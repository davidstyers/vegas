"""Data Layer for the Vegas backtesting engine.

This module provides functionality for ingesting and querying historical market data
using polars and numpy for efficient vectorized operations.
"""

from typing import Dict, List, Optional, Union, Set
from datetime import datetime, timedelta
import os
import polars as pl
import numpy as np
import glob
import zstandard as zstd
import io
import logging
from pathlib import Path
import tempfile
import shutil
import pytz
from collections import OrderedDict

from vegas.database import DatabaseManager


class DataLayer:
    """Data layer for the Vegas backtesting engine."""
    
    def __init__(self, data_dir: str = "db", test_mode: bool = False, timezone: str = "EST"):
        """Initialize the DataLayer.
        
        Args:
            data_dir: Directory for storing data files
            test_mode: If True, uses an isolated in-memory database for testing
            timezone: Timezone for timestamp conversion (default: 'EST')
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger('vegas.data')
        self.data = None  # Main DataFrame with all market data
        self.symbols = set()  # Set of available symbols
        self.date_range = None  # Tuple of (start_date, end_date)
        self.timezone = timezone  # Store the timezone for timestamp conversions
        
        # Validate timezone
        try:
            pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.warning(f"Unknown timezone: {timezone}, falling back to EST")
            self.timezone = "EST"
            
        self.logger.info(f"DataLayer initialized with timezone: {self.timezone}")
        
        # Check for environment variable to enforce test mode
        if os.environ.get('VEGAS_TEST_MODE') == '1':
            test_mode = True
            self.logger.info("Test mode enforced by VEGAS_TEST_MODE environment variable")
            
        self.test_mode = test_mode
        
        # Create necessary directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database manager
        if test_mode:
            # For tests, use in-memory database to avoid polluting real data
            self.db_path = ":memory:"
            self.parquet_dir = tempfile.mkdtemp(prefix="vegas_test_")
            self.logger.info(f"Test mode enabled: Using in-memory database and temporary directory {self.parquet_dir}")
        else:
            # For normal operation, use the specified data directory
            self.db_path = os.path.join(data_dir, "vegas.duckdb")
            self.parquet_dir = data_dir
            
        self.db_manager = None
        self.use_database = True
        
        # ---- Cache configuration ----
        self._cache_enabled = True
        self._query_cache_max = 64
        self._slice_cache_max = 16
        self._query_cache: "OrderedDict[tuple, pl.DataFrame]" = OrderedDict()
        # maps (date_iso, frozenset(symbols), market_hours_tuple, timezone) -> dict[datetime, pl.DataFrame]
        self._slice_cache: "OrderedDict[tuple, Dict[datetime, pl.DataFrame]]" = OrderedDict()
        # -----------------------------------
        
        try:
            self.db_manager = DatabaseManager(self.db_path, self.parquet_dir)
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize database: {e}")
            self.use_database = False
        self.engine = None
    
    def __del__(self):
        """Clean up any temporary resources."""
        self.close()
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'parquet_dir') and self.parquet_dir.startswith(tempfile.gettempdir()):
            try:
                # Clean up temporary directory if it exists
                if os.path.exists(self.parquet_dir):
                    shutil.rmtree(self.parquet_dir)
                    self.logger.info(f"Cleaned up temporary test directory: {self.parquet_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up test directory {self.parquet_dir}: {e}")
    
    def _convert_timestamp_timezone(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert dataframe timestamps to the configured timezone.
        
        Args:
            df: DataFrame containing a 'timestamp' column
            
        Returns:
            DataFrame with timestamps converted to the configured timezone
        """
        if df is None or df.is_empty() or 'timestamp' not in df.columns:
            return df
            
        return df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", time_zone=self.timezone)))

    # ---- Cache helpers ----
    def _make_query_key(self, start: datetime, end: datetime, symbols: Optional[List[str]], market_hours: Optional[tuple]) -> tuple:
        syms = tuple(sorted(symbols)) if symbols else None
        mh = tuple(market_hours) if market_hours else None
        return (start.isoformat(), end.isoformat(), syms, mh, str(self.timezone))

    def _lru_get(self, cache: OrderedDict, key):
        if key in cache:
            val = cache.pop(key)
            cache[key] = val
            return val
        return None

    def _lru_set(self, cache: OrderedDict, key, val, maxsize: int):
        if key in cache:
            cache.pop(key)
        cache[key] = val
        while len(cache) > maxsize:
            cache.popitem(last=False)
    # ------------------------
    
    def is_initialized(self) -> bool:
        """Check if the data layer is initialized with data."""
        # Check if in-memory data is available
        if self.data is not None and not self.data.is_empty():
            return True
            
        # Check if database data is available
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                return not dates.is_empty() and dates.row(0)[dates.columns.index('day_count')] > 0
            except Exception as e:
                self.logger.warning(f"Database query failed: {e}")
                
        return False
    
    def load_data(self, file_path: str = None, directory: str = None, file_pattern: str = "*.csv*", max_files: int = None) -> None:
        """Load market data from a file or directory.
        
        Args:
            file_path: Path to a CSV or compressed CSV file
            directory: Directory containing data files
            file_pattern: Pattern for matching files
        """
        if file_path:
            self._load_single_file(file_path)
        elif directory:
            self._load_multiple_files(directory, file_pattern, max_files)
        else:
            # Try to load from database if no file specified
            self._try_load_from_database()
    
    def _try_load_from_database(self) -> None:
        """Try to load data from database if available."""
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                if not dates.is_empty() and dates.select(pl.col('day_count')).item() > 0:
                    self.logger.info("Using data from database")
                    # Load a sample to initialize the data layer
                    data = self.db_manager.get_market_data(
                        start_date=dates.select('start_date').item(),
                        end_date=dates.select('start_date').item() + timedelta(days=1)
                    )
                    if not data.is_empty():
                        self._validate_and_process_data(data)
                        return
            except Exception as e:
                self.logger.warning(f"Failed to load data from database: {e}")
        
        raise ValueError("No data source specified and no data available in database")
    
    def _load_single_file(self, file_path: str) -> None:
        """Load market data from a single file."""
        self.logger.info(f"Loading data from file: {file_path}")
        
        # Handle OHLCV files for database ingestion
        if file_path.endswith('.ohlcv-1h.csv.zst') and self.use_database and self.db_manager:
            try:
                self.db_manager.ingest_ohlcv_file(file_path)
                self._try_load_from_database()
                return
            except Exception as e:
                self.logger.error(f"Failed to ingest OHLCV file: {e}")
        
        # Load the file into a DataFrame
        df = self._read_file(file_path)
        
        # Process the data
        self._validate_and_process_data(df)
        
        # If database is available, also ingest into database
        if self.use_database and self.db_manager:
            try:
                source_name = os.path.basename(file_path)
                self.db_manager.ingest_data(df, source_name)
            except Exception as e:
                self.logger.error(f"Failed to ingest data into database: {e}")
    
    def _read_file(self, file_path: str) -> pl.DataFrame:
        """Read file and return DataFrame."""
        if file_path.endswith('.zst'):
            # Decompress Zstandard file
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                data_buffer = dctx.decompress(f.read())
                csv_data = io.BytesIO(data_buffer)
                df = pl.read_csv(csv_data, try_parse_dates=True)
        else:
            # Regular CSV file
            df = pl.scan_csv(file_path, try_parse_dates=True)
            
        # Convert timestamps to the configured timezone
        return self._convert_timestamp_timezone(df)
    
    def _load_multiple_files(self, directory: str = None, file_pattern: str = "*.csv*", 
                           max_files: int = None) -> None:
        """Load market data from multiple files."""
        if directory is None:
            directory = self.data_dir
            
        self.logger.info(f"Loading data from directory: {directory} with pattern: {file_pattern}")
        
        # Handle OHLCV directory for database ingestion
        if file_pattern == "*.ohlcv-1h.csv.zst" and self.use_database and self.db_manager:
            try:
                self.db_manager.ingest_ohlcv_directory(directory, file_pattern, max_files)
                self._try_load_from_database()
                return
            except Exception as e:
                self.logger.error(f"Failed to ingest OHLCV files: {e}")
        
        # Find data files
        files = self._find_files(directory, file_pattern)
        
        if not files:
            raise FileNotFoundError(f"No data files found in {directory} with pattern {file_pattern}")
            
        # Limit the number of files if specified
        if max_files:
            files = files[:max_files]
            
        self.logger.info(f"Found {len(files)} data files to load")
        
        # Load all files into a list of dataframes
        dfs = []
        for file in files:
            try:
                df = self._read_file(file)
                dfs.append(df)
                
                # If database is available, also ingest each file
                if self.use_database and self.db_manager:
                    try:
                        source_name = os.path.basename(file)
                        self.db_manager.ingest_data(df, source_name)
                    except Exception as e:
                        self.logger.error(f"Failed to ingest {file} into database: {e}")
                
            except Exception as e:
                self.logger.error(f"Error loading file {file}: {e}")
                
        if not dfs:
            raise ValueError("No valid data files were loaded")
            
        # Combine all dataframes
        combined_df = pl.concat(dfs)
        self._validate_and_process_data(combined_df)
    
    def _find_files(self, directory: str, file_pattern: str) -> List[str]:
        """Find files matching a pattern in a directory."""
        search_path = os.path.join(directory, file_pattern)
        files = sorted(glob.glob(search_path))
        
        # Try subdirectories if no files found
        if not files:
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                if os.path.isdir(subdir_path):
                    search_path = os.path.join(subdir_path, file_pattern)
                    subdir_files = sorted(glob.glob(search_path))
                    files.extend(subdir_files)
                    
        return files
    
    def _validate_and_process_data(self, df: pl.DataFrame) -> None:
        """Validate and process loaded data."""
        # Check if dataframe is empty
        if df.is_empty():
            raise ValueError("Data is empty")
            
        # Check required columns
        required_cols = ['timestamp', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")
            
        # Sort by timestamp
        df = df.sort('timestamp')
        
        # Ensure timestamp is a datetime with proper timezone
        if df.schema['timestamp'].dtype != pl.Datetime:
            df = df.with_columns(pl.col('timestamp').cast(pl.Datetime))
            
        # Apply timezone conversion
        df = self._convert_timestamp_timezone(df)
        
        # Store data in memory
        self.data = df
        
        # Update derived properties
        self.symbols = set(df.select('symbol').unique().to_series().to_list())
        self.date_range = (df.select('timestamp').min().item(), df.select('timestamp').max().item())
        
        self.logger.info(f"Processed {len(df)} rows with {len(self.symbols)} symbols")
    
    def get_unified_timestamp_index(self, start_ts: datetime, end_ts: datetime) -> pl.Series:
        """Return unique, sorted timestamps across all data sources within [start_ts, end_ts].

        This delegates to the database manager which queries the canonical union view.
        The returned series is timezone-aware in the configured timezone.
        """
        if not self.use_database or self.db_manager is None:
            # Fallback: derive from in-memory data if present
            if self.data is None or self.data.is_empty():
                return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=self.timezone))
            start_bound = start_ts
            end_bound = end_ts
            try:
                filtered = self.data.filter(
                    (pl.col("timestamp") >= pl.lit(start_bound).cast(pl.Datetime("us", self.timezone))) &
                    (pl.col("timestamp") <= pl.lit(end_bound).cast(pl.Datetime("us", self.timezone)))
                )
            except Exception:
                # Best-effort without explicit cast if schema already matches
                filtered = self.data.filter((pl.col("timestamp") >= start_bound) & (pl.col("timestamp") <= end_bound))
            if filtered.is_empty():
                return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=self.timezone))
            ts = filtered.select(pl.col("timestamp")).unique().sort("timestamp").get_column("timestamp")
            # Ensure timezone
            return ts.cast(pl.Datetime("us", time_zone=self.timezone))

        try:
            return self.db_manager.get_unified_timestamps(start_ts, end_ts, timezone=self.timezone)
        except Exception as e:
            self.logger.error(f"get_unified_timestamp_index failed: {e}")
            return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=self.timezone))

    def get_data_for_backtest(self, start: datetime, end: datetime, symbols: List[str] = None, market_hours: tuple = None) -> pl.DataFrame:
        """Get data for a backtest period.
        
        Args:
            start: Start date
            end: End date
            symbols: Optional list of symbols to include
            
        Returns:
            DataFrame with market data
        """
        # Ensure start and end are timezone-aware
        start_ts = start
        end_ts = end

        def _time_str_to_minutes(time_str: str) -> int:
            hour, minute = map(int, time_str.split(":"))
            return hour * 60 + minute
        
        # Try cache first
        if self._cache_enabled:
            qkey = self._make_query_key(start_ts, end_ts, symbols, market_hours)
            cached = self._lru_get(self._query_cache, qkey)
            if cached is not None:
                self.data = cached
                return cached

        # Try to get data from database first if available
        if self.use_database and self.db_manager:
            try:
                raw = self.db_manager.get_market_data(start_date=start_ts, end_date=end_ts, symbols=symbols, timezone=self.timezone)
                if not raw.is_empty():
                    if market_hours:
                        start_m = _time_str_to_minutes(market_hours[0])
                        end_m = _time_str_to_minutes(market_hours[1])
                        result = raw.filter(
                            (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int32)).is_between(start_m, end_m, closed="left")
                        )
                    else:
                        result = raw
                    self.data = result
                    if self._cache_enabled:
                        self._lru_set(self._query_cache, qkey, result, self._query_cache_max)
                    return result
            finally:
                self.logger.info("Data loaded from database")
            
        # No data available
        return pl.DataFrame()
    
    def get_unique_dates(self) -> List[datetime]:
        """Get unique dates in the dataset."""
        if self.data is None:
            if self.use_database and self.db_manager:
                try:
                    dates = self.db_manager.get_available_dates()
                    if not dates.is_empty():
                        start_date = dates.select(pl.col('start_date')).item()
                        end_date = dates.select(pl.col('end_date')).item()
                        delta = (end_date - start_date).days + 1
                        return [(start_date + timedelta(days=i)).date() for i in range(delta)]
                except Exception:
                    pass
            return []
    
    def get_all_trading_days(self) -> List[datetime]:
        """Get all trading days in the dataset."""
        if self.use_database and self.db_manager:
            return self.db_manager.get_available_trading_days()
        return []
    
    def get_universe(self, date: datetime.date = None) -> List[str]:
        """Get the universe of available symbols.
        
        Args:
            date: Optional date to filter symbols by availability
            
        Returns:
            List of symbols
        """
        if date is None:
            return sorted(list(self.symbols)) if self.symbols else []
            
        # Filter data by date and return unique symbols
        if self.data is not None and not self.data.is_empty():
            try:
                # Get the date portion only
                date_floor = datetime.combine(date, datetime.min.time())
                date_ceil = datetime.combine(date, datetime.max.time())
                date_floor, date_ceil = date_floor.astimezone(pytz.timezone(self.timezone)), date_ceil.astimezone(pytz.timezone(self.timezone))

                filtered_data = self.data.filter(
                    (pl.col("timestamp") >= pl.lit(date_floor).cast(pl.Datetime("us", self.timezone))) &
                    (pl.col("timestamp") <= pl.lit(date_ceil).cast(pl.Datetime("us", self.timezone)))
                )
                if not filtered_data.is_empty():
                    symbols = filtered_data.select('symbol').unique().to_series().to_list()
                    return symbols
                else:
                    self.logger.warning(f"No data available for date {date}")
                    # If no data for the specific date, try to get data from nearby dates
                    window_size = 5  # Look 5 days before and after
                    expanded_start = date_floor - timedelta(days=window_size)
                    expanded_end = date_ceil + timedelta(days=window_size)
                    
                    nearby_data = self.data.filter(
                        (pl.col('timestamp') >= expanded_start) & 
                        (pl.col('timestamp') <= expanded_end)
                    )
                    
                    if not nearby_data.is_empty():
                        self.logger.info(f"Using symbols from nearby dates for {date}")
                        symbols = nearby_data.select('symbol').unique().to_series().to_list()
                        return symbols
            except Exception as e:
                self.logger.error(f"Error getting universe for date {date}: {e}")
        
        # If we get here, we couldn't find any symbols for the date
        self.logger.warning(f"Falling back to all known symbols for date {date}")
        return list(self.symbols) if self.symbols else []
    
    def get_available_date_range(self) -> tuple:
        """Get the available date range in the dataset."""
        if self.date_range:
            return self.date_range
            
        # Try to get from database
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                if not dates.is_empty():
                    return (dates.select('start_date').item(), dates.select('end_date').item())
            except Exception:
                pass
                
        # Use data if available
        if self.data is not None and not self.data.is_empty():
            return (self.data.select(pl.col('timestamp').min()).item(), self.data.select(pl.col('timestamp').max()).item())
            
        return (None, None)
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data."""
        start_date, end_date = self.get_available_date_range()
        
        if start_date is None or end_date is None:
            return {
                "symbol_count": 0,
                "start_date": None,
                "end_date": None,
                "days": 0
            }
            
        # Calculate days
        if isinstance(start_date, datetime) and isinstance(end_date, datetime):
            days = (end_date - start_date).days + 1
        else:
            days = 0
            
        # Get row and symbol counts
        if self.data is not None:
            symbol_count = len(self.symbols)
        elif self.use_database and self.db_manager:
            try:
                # Get from database
                symbols = self.db_manager.get_available_symbols()
                symbol_count = len(symbols)
            except Exception:
                symbol_count = 0
        else:
            symbol_count = 0
            
        return {
            "symbol_count": symbol_count,
            "start_date": start_date,
            "end_date": end_date,
            "days": days
        }
    
    def ingest_to_database(self, df: pl.DataFrame, source_name: str) -> int:
        """Ingest data to the database."""
        if not self.use_database or self.db_manager is None:
            self.logger.warning("Database not available for ingestion")
            return 0
            
        try:
            rows_affected = self.db_manager.ingest_data(df, source_name)
            self.logger.info(f"Ingested {rows_affected} rows into database")
            return rows_affected
        except Exception as e:
            self.logger.error(f"Database ingestion failed: {e}")
            return 0
    
    def ingest_ohlcv_file(self, file_path: str) -> int:
        """Ingest an OHLCV file into the database."""
        if not self.use_database or self.db_manager is None:
            self.logger.warning("Database not available for ingestion")
            return 0
            
        try:
            rows_affected = self.db_manager.ingest_ohlcv_file(file_path)
            self.logger.info(f"Ingested {rows_affected} rows from OHLCV file")
            return rows_affected
        except Exception as e:
            self.logger.error(f"OHLCV file ingestion failed: {e}")
            return 0
    
    def ingest_ohlcv_directory(self, directory: str, max_files: int = None) -> int:
        """Ingest OHLCV files from a directory into the database."""
        if not self.use_database or self.db_manager is None:
            self.logger.warning("Database not available for ingestion")
            return 0
            
        try:
            rows_affected = self.db_manager.ingest_ohlcv_directory(
                directory=directory, 
                pattern="*.ohlcv-1h.csv.zst", 
                max_files=max_files
            )
            self.logger.info(f"Ingested {rows_affected} rows from OHLCV directory")
            return rows_affected
        except Exception as e:
            self.logger.error(f"OHLCV directory ingestion failed: {e}")
            return 0
    
    def close(self) -> None:
        """Close database connections."""
        if self.use_database and self.db_manager:
            try:
                self.db_manager.close()
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
    
    def get_data_for_timestamp(self, timestamp: datetime) -> pl.DataFrame:
        """Get market data for a specific timestamp.
        
        Args:
            timestamp: The timestamp to get data for
            
        Returns:
            DataFrame with market data for the timestamp
        """
        # Ensure timestamp is timezone-aware
        ts = timestamp
        if ts.tz is None:
            ts = ts.tz_localize(self.timezone)
        elif str(ts.tz) != str(self.timezone):  # Compare string representations instead of .zone attribute
            try:
                ts = ts.tz_convert(self.timezone)
            except Exception as e:
                self.logger.warning(f"Failed to convert timestamp timezone: {e}")
            
        # Try to get data from database first if available
        if self.use_database and self.db_manager:
            try:
                # Use a small window around the timestamp to ensure we get data
                start_time = ts - timedelta(minutes=1)
                end_time = ts + timedelta(minutes=1)
                data = self.db_manager.get_market_data(start_date=start_time, end_date=end_time)
                
                # Filter to get closest timestamps
                if not data.is_empty():
                    # Convert timestamps to the configured timezone
                    data = self._convert_timestamp_timezone(data)
                    
                    # Get the closest timestamp for each symbol
                    grouped = data.groupby('symbol')
                    closest_data = []
                    
                    for symbol, group in grouped:
                        # Find the row with timestamp closest to the target
                        group['time_diff'] = abs(group['timestamp'] - ts)
                        closest_row = group.loc[group['time_diff'].idxmin()].drop('time_diff')
                        closest_data.append(closest_row)
                    
                    if closest_data:
                        return pl.DataFrame(closest_data) if closest_data else pl.DataFrame()
            except Exception as e:
                self.logger.warning(f"Failed to get data from database for timestamp {ts}: {e}")
        
        # Fall back to in-memory data if available
        if self.data is not None:
            # Make sure timestamps have timezone info for comparison
            data_copy = self.data.clone()
            dtype = data_copy.schema['timestamp']
            if isinstance(dtype, pl.datatypes.Datetime):
                current_tz = dtype.tz
            else:
                current_tz = None

            if current_tz is None:
                # Localize naive timestamps to UTC then convert
                data_copy = data_copy.with_columns(
                    pl.col('timestamp').cast(pl.Datetime('us', time_zone='UTC')).cast(pl.Datetime('us', time_zone=self.timezone)))
            else:
                 # If timezone differs, convert
                if current_tz != self.timezone:
                    try:
                        data_copy = data_copy.with_columns(
                        pl.col('timestamp').cast(pl.Datetime('us', time_zone=self.timezone)))
                    except Exception as e:
                        self.logger.warning(f"Failed to convert timestamp timezone: {e}")
                
            # Find the closest timestamp for each symbol
            data_copy['time_diff'] = abs(data_copy['timestamp'] - ts)
            
            # Get the closest timestamp for each symbol
            closest_idx = data_copy.groupby('symbol')['time_diff'].idxmin()
            result = data_copy.loc[closest_idx].drop('time_diff', axis=1)
            
            return result
            
        # No data available
        return pl.DataFrame()
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> pl.Series:
        """Return unique trading days (pl.Date) within [start_date, end_date] that have data.
        """
        # No data available
        if self.data is None or self.data.is_empty():
            return pl.Series("date", [], dtype=pl.Date)

        # Normalize bounds to polars Datetime in configured timezone (eager literals)
        start_ts = start_date.astimezone(pytz.timezone(self.timezone))
        end_ts = end_date.astimezone(pytz.timezone(self.timezone))

        filtered = self.data.filter(
                (pl.col("timestamp") >= pl.lit(start_ts).cast(pl.Datetime("us", self.timezone))) &
                (pl.col("timestamp") <= pl.lit(end_ts).cast(pl.Datetime("us", self.timezone)))
            )

        if filtered.is_empty():
            return pl.Series("date", [], dtype=pl.Date)

        # Extract unique dates and return as Series
        dates_df = (
            filtered
            .select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .sort("date")
        )
        return dates_df.get_column("date")
    
    def get_data_for_date(self, date: datetime) -> pl.DataFrame:
        """Get all market data for a specific date.
        
        Args:
            date: The date to get data for
            
        Returns:
            DataFrame with market data for the date
        """
        date_floor = datetime.combine(date.date(), datetime.min.time())
        date_ceil = datetime.combine(date.date(), datetime.max.time())
        
        return self.get_data_for_backtest(date_floor, date_ceil)

    def history(self, assets: Union[str, List[str]], current_dt: datetime, fields: Union[str, List[str]], bar_count: int, frequency: str = '1h') -> pl.DataFrame:
        """
        Fetch a window of historical market data.

        This method depends on the engine attribute being set on this DataLayer instance,
        which provides access to the current simulation time via `self.engine.context.current_dt`.

        Args:
            assets: A single symbol string or a list of symbol strings.
            fields: A single field string (e.g., 'price') or a list of field strings.
            bar_count: The number of historical bars to retrieve.
            frequency: The data frequency (e.g., '1h', '1d').

        Returns:
            A Polars DataFrame containing the requested historical data, indexed by datetime
            with columns for each asset and field.
        """

        # 1. Normalize parameters
        if isinstance(assets, str):
            assets = [assets]
        if isinstance(fields, str):
            fields = [fields]
        
        # Handle 'price' as a common alias for 'close'
        if 'price' in fields:
            fields = ['close' if f == 'price' else f for f in fields]

        # 2. Construct and execute the query
        # We need to select the `bar_count` most recent bars for each asset before `current_dt`.
        query_fields = sorted(list(set(fields + ['timestamp', 'symbol'])))
        fields_str = ", ".join(f'"{f}"' for f in query_fields)
        assets_str = ", ".join([f"'{s}'" for s in assets])

        query = f"""
        WITH ranked_data AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
            FROM market_data
            WHERE
                symbol IN ({assets_str})
                AND timestamp <= ?
        )
        SELECT {fields_str}
        FROM ranked_data
        WHERE rn <= ?
        ORDER BY timestamp ASC, symbol
        """
        
        params = (current_dt, bar_count)
        self.logger.debug(f"Executing history query with params {params}")
        
        try:
            df = self.db_manager.query_to_df(query, params)
        except Exception as e:
            self.logger.error(f"Failed to fetch history: {e}")
            return pl.DataFrame()

        if df.is_empty():
            return pl.DataFrame()

        # 3. Handle resampling if requested frequency is different
        # Assuming native frequency is 1h based on ingestion methods.
        native_frequency = '1h'
        if frequency != native_frequency:
            self.logger.info(f"Resampling data from '{native_frequency}' to '{frequency}'")
            
            agg_exprs = []
            if 'open' in fields: agg_exprs.append(pl.col('open').first().alias('open'))
            if 'high' in fields: agg_exprs.append(pl.col('high').max().alias('high'))
            if 'low' in fields: agg_exprs.append(pl.col('low').min().alias('low'))
            if 'close' in fields: agg_exprs.append(pl.col('close').last().alias('close'))
            if 'volume' in fields: agg_exprs.append(pl.col('volume').sum().alias('volume'))

            if not agg_exprs:
                self.logger.warning(f"Resampling for fields {fields} is not defined, returning un-resampled data.")
            else:
                df = df.sort("timestamp").group_by_dynamic(
                    "timestamp",
                    every=frequency,
                    by="symbol"
                ).agg(agg_exprs)

        # 4. Pivot to wide format as requested
        if not df.is_empty():
            try:
                # This creates columns like 'close_AAPL', 'volume_AAPL'
                pivot_values = [f for f in fields if f in df.columns]
                if pivot_values:
                    df = df.pivot(index='timestamp', columns='symbol', values=pivot_values)
            except Exception as e:
                self.logger.warning(f"Could not pivot data to wide format: {e}. Returning long format.")

        return df
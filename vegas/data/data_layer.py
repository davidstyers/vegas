"""Data Layer for the Vegas backtesting engine.

This module provides functionality for ingesting and querying historical market data
using polars and numpy for efficient vectorized operations.
"""

from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import os
import polars as pl
import glob
import zstandard as zstd
import io
import logging
import tempfile
import shutil
import pytz

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
        
        # Note: All runtime caching has moved to DataPortal. DataLayer performs IO/DB only.
        
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

    # Cache helpers removed. DataLayer no longer implements an internal cache.
    
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
            ts = self.db_manager.get_unified_timestamps(start_ts, end_ts, timezone=self.timezone)
            return ts
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
                    return result
            finally:
                self.logger.debug("Data loaded from database")
            
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
                    out = (dates.select('start_date').item(), dates.select('end_date').item())
                    return out
            except Exception:
                pass
                
        # Use data if available
        if self.data is not None and not self.data.is_empty():
            out = (self.data.select(pl.col('timestamp').min()).item(), self.data.select(pl.col('timestamp').max()).item())
            return out
            
        return (None, None)
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data."""
        # Caching is handled by subordinate methods; build response cheaply
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
    
    def get_data_for_timestamp(self, timestamp: datetime, symbols: Optional[Union[List[str], str]] = None, market_hours: Optional[tuple] = None) -> pl.DataFrame:
        """Return rows for the exact timestamp, optionally filtered by symbols.
        """
        # Normalize symbols
        if isinstance(symbols, str):
            sym_list: Optional[List[str]] = [symbols]
        else:
            sym_list = symbols

        # Query only the specific instant via backtest path to leverage its cache
        df = self.get_data_for_backtest(start=timestamp, end=timestamp, symbols=sym_list, market_hours=market_hours)
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df
    
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
        out = dates_df.get_column("date")
        return out
    
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

    def history(self, assets: Union[str, List[str]], current_dt: datetime, fields: Optional[Union[str, List[str]]] = None, bar_count: int = 1, frequency: str = '1h') -> pl.DataFrame:
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
        # Default fields
        if not fields:
            fields = ['open', 'high', 'low', 'close', 'volume']

        # Normalize parameters
        symbols: Optional[List[str]]
        if isinstance(assets, str):
            symbols = [assets]
        else:
            symbols = assets
        if isinstance(fields, str):
            fields = [fields]

        # Handle 'price' alias
        if 'price' in fields:
            fields = ['close' if f == 'price' else f for f in fields]

        # Retrieve bars via DatabaseManager to avoid duplicated SQL logic here
        try:
            df = self.db_manager.get_market_data(
                end_date=current_dt,
                symbols=symbols,
                timezone=self.timezone,
                bar_count=bar_count,
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch history via DatabaseManager: {e}")
            return pl.DataFrame()

        if df.is_empty():
            return pl.DataFrame()

        # Select requested fields plus timestamp/symbol to keep consistent
        select_cols = ['timestamp', 'symbol'] + [f for f in fields if f in df.columns]
        df = df.select(select_cols)

        # Resample if needed (native frequency assumed '1h')
        native_frequency = '1h'
        if frequency != native_frequency:
            self.logger.info(f"Resampling data from '{native_frequency}' to '{frequency}'")

            agg_exprs = []
            if 'open' in fields and 'open' in df.columns:
                agg_exprs.append(pl.col('open').first().alias('open'))
            if 'high' in fields and 'high' in df.columns:
                agg_exprs.append(pl.col('high').max().alias('high'))
            if 'low' in fields and 'low' in df.columns:
                agg_exprs.append(pl.col('low').min().alias('low'))
            if 'close' in fields and 'close' in df.columns:
                agg_exprs.append(pl.col('close').last().alias('close'))
            if 'volume' in fields and 'volume' in df.columns:
                agg_exprs.append(pl.col('volume').sum().alias('volume'))

            if agg_exprs:
                df = (
                    df.sort("timestamp")
                      .group_by_dynamic("timestamp", every=frequency, by="symbol")
                      .agg(agg_exprs)
                )
            else:
                self.logger.warning(f"Resampling for fields {fields} is not defined; returning un-resampled data.")

        # Pivot to wide format only when a specific symbol subset was requested
        if not df.is_empty() and symbols:
            try:
                pivot_values = [f for f in fields if f in df.columns]
                if pivot_values:
                    df = df.pivot(index='timestamp', columns='symbol', values=pivot_values)
            except Exception as e:
                self.logger.warning(f"Could not pivot data to wide format: {e}. Returning long format.")

        return df
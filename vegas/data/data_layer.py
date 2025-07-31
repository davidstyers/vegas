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

from vegas.database import DatabaseManager


class DataLayer:
    """Minimal, vectorized data layer for the Vegas backtesting engine."""
    
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
        
        try:
            self.db_manager = DatabaseManager(self.db_path, self.parquet_dir)
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize database: {e}")
            self.use_database = False
    
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
                data = self.db_manager.get_market_data(start_date=start_ts, end_date=end_ts, symbols=symbols, timezone=self.timezone)
                if not data.is_empty():
                    if market_hours:
                        start = _time_str_to_minutes(market_hours[0])
                        end = _time_str_to_minutes(market_hours[1])
                        return data.filter(
                            (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60 + pl.col("timestamp").dt.minute().cast(pl.Int32)).is_between(start, end, closed="left")
                        )
                    else:
                        return data
            except Exception as e:
                self.logger.warning(f"Failed to get data from database: {e}")
            
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
            
        # Get unique dates from timestamp column
        return self.data.select(
            pl.col('timestamp').dt.date().alias('date')
        ).unique().sort('date').get_column('date').to_list()
    
    def get_universe(self, date: datetime = None) -> List[str]:
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
                date_floor = datetime.combine(date.date(), datetime.min.time())
                date_ceil = datetime.combine(date.date(), datetime.max.time())
                
                filtered_data = self.data.filter(
                    (pl.col('timestamp') >= date_floor) & 
                    (pl.col('timestamp') <= date_ceil)
                )
                
                if not filtered_data.is_empty():
                    symbols = filtered_data.select('symbol').unique().to_series().to_list()
                    return sorted(symbols)
                else:
                    self.logger.warning(f"No data available for date {date.date()}")
                    # If no data for the specific date, try to get data from nearby dates
                    window_size = 5  # Look 5 days before and after
                    expanded_start = date_floor - timedelta(days=window_size)
                    expanded_end = date_ceil + timedelta(days=window_size)
                    
                    nearby_data = self.data.filter(
                        (pl.col('timestamp') >= expanded_start) & 
                        (pl.col('timestamp') <= expanded_end)
                    )
                    
                    if not nearby_data.is_empty():
                        self.logger.info(f"Using symbols from nearby dates for {date.date()}")
                        symbols = nearby_data.select('symbol').unique().to_series().to_list()
                        return sorted(symbols)
            except Exception as e:
                self.logger.error(f"Error getting universe for date {date}: {e}")
        
        # If we get here, we couldn't find any symbols for the date
        self.logger.warning(f"Falling back to all known symbols for date {date}")
        return sorted(list(self.symbols)) if self.symbols else []
    
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
        
        # if start_date is None or end_date is None:
        #     return {
        #         "symbol_count": 0,
        #         "start_date": None,
        #         "end_date": None,
        #         "days": 0
        #     }
            
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
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get all trading days in the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DatetimeIndex with trading days
        """
        # Ensure dates are timezone-aware
        start_ts = start_date
        end_ts = end_date
        
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(self.timezone)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(self.timezone)
            
        # First try to get data from the current dataset
        if self.data is not None:
            # Make sure timestamps have timezone info
            data_with_tz = self.data.clone()
            data_timestamps = data_with_tz['timestamp']
            
            if data_timestamps.dt.tz is None:
                # Localize to UTC first, then convert to target timezone
                data_with_tz['timestamp'] = data_timestamps.dt.tz_localize('UTC')
                try:
                    data_with_tz['timestamp'] = data_with_tz['timestamp'].dt.tz_convert(self.timezone)
                except Exception as e:
                    self.logger.warning(f"Failed to convert timestamp timezone: {e}")
            else:
                # Compare string representations instead of .zone attribute
                current_tz_str = str(data_timestamps.dt.tz)
                target_tz_str = str(self.timezone)
                if current_tz_str != target_tz_str:
                    try:
                        data_with_tz['timestamp'] = data_timestamps.dt.tz_convert(self.timezone)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert timestamp timezone: {e}")
                
            # Filter by date range and extract unique dates
            filtered_data = data_with_tz[
                (data_with_tz['timestamp'] >= start_ts) & 
                (data_with_tz['timestamp'] <= end_ts)
            ]
            
            if not filtered_data.is_empty():
                # Extract unique dates (just the date part, not time)
                # Get unique dates from timestamp
                dates = filtered_data.select(
                    pl.col('timestamp').dt.date().alias('date')).unique().sort('date').get_column('date').to_list()
        return dates
    
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
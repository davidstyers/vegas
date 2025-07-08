"""Data Layer for the Vegas backtesting engine.

This module provides functionality for ingesting and querying historical market data
using pandas and numpy for efficient vectorized operations.
"""

from typing import Dict, List, Optional, Union, Set
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import glob
import zstandard as zstd
import io
import logging
from pathlib import Path

from vegas.database import DatabaseManager


class DataLayer:
    """Minimal, vectorized data layer for the Vegas backtesting engine."""
    
    def __init__(self, data_dir: str = "db"):
        """Initialize the DataLayer.
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger('vegas.data')
        self.data = None  # Main DataFrame with all market data
        self.symbols = set()  # Set of available symbols
        self.date_range = None  # Tuple of (start_date, end_date)
        
        # Create necessary directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database manager
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
    
    def is_initialized(self) -> bool:
        """Check if the data layer is initialized with data."""
        # Check if in-memory data is available
        if self.data is not None and not self.data.empty:
            return True
            
        # Check if database data is available
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                return not dates.empty and dates['day_count'].iloc[0] > 0
            except Exception as e:
                self.logger.warning(f"Database query failed: {e}")
                
        return False
    
    def load_data(self, file_path: str = None, directory: str = None, file_pattern: str = "*.csv*") -> None:
        """Load market data from a file or directory.
        
        Args:
            file_path: Path to a CSV or compressed CSV file
            directory: Directory containing data files
            file_pattern: Pattern for matching files
        """
        if file_path:
            self._load_single_file(file_path)
        elif directory:
            self._load_multiple_files(directory, file_pattern)
        else:
            # Try to load from database if no file specified
            self._try_load_from_database()
    
    def _try_load_from_database(self) -> None:
        """Try to load data from database if available."""
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                if not dates.empty and dates['day_count'].iloc[0] > 0:
                    self.logger.info("Using data from database")
                    # Load a sample to initialize the data layer
                    data = self.db_manager.get_market_data(
                        start_date=dates['start_date'].iloc[0],
                        end_date=dates['start_date'].iloc[0] + pd.Timedelta(days=1)
                    )
                    if not data.empty:
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
    
    def _read_file(self, file_path: str) -> pd.DataFrame:
        """Read file and return DataFrame."""
        if file_path.endswith('.zst'):
            # Decompress Zstandard file
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                data_buffer = dctx.decompress(f.read())
                csv_data = io.StringIO(data_buffer.decode('utf-8'))
                return pd.read_csv(csv_data, parse_dates=['timestamp'])
        else:
            # Regular CSV file
            return pd.read_csv(file_path, parse_dates=['timestamp'])
    
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
        combined_df = pd.concat(dfs, ignore_index=True)
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
    
    def _validate_and_process_data(self, df: pd.DataFrame) -> None:
        """Validate and process loaded data."""
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("Data is empty")
            
        # Check required columns
        required_cols = ['timestamp', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")
            
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Ensure timestamp is a datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Store data in memory
        self.data = df
        
        # Update derived properties
        self.symbols = set(df['symbol'].unique())
        self.date_range = (df['timestamp'].min(), df['timestamp'].max())
        
        self.logger.info(f"Processed {len(df)} rows with {len(self.symbols)} symbols")
    
    def get_data_for_backtest(self, start: datetime, end: datetime, symbols: List[str] = None) -> pd.DataFrame:
        """Get data for a backtest period.
        
        Args:
            start: Start date
            end: End date
            symbols: Optional list of symbols to include
            
        Returns:
            DataFrame with market data
        """
        # Try to get data from database first if available
        if self.use_database and self.db_manager:
            try:
                data = self.db_manager.get_market_data(start_date=start, end_date=end, symbols=symbols)
                if not data.empty:
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to get data from database: {e}")
        
        # Fall back to in-memory data if available
        if self.data is not None:
            # Filter by date range
            filtered_data = self.data[
                (self.data['timestamp'] >= pd.Timestamp(start)) & 
                (self.data['timestamp'] <= pd.Timestamp(end))
            ]
            
            # Filter by symbols if provided
            if symbols:
                filtered_data = filtered_data[filtered_data['symbol'].isin(symbols)]
                
            return filtered_data
            
        # No data available
        return pd.DataFrame()
    
    def get_unique_dates(self) -> pd.DatetimeIndex:
        """Get unique dates in the dataset."""
        if self.data is None:
            if self.use_database and self.db_manager:
                try:
                    dates = self.db_manager.get_available_dates()
                    if not dates.empty:
                        return pd.DatetimeIndex(pd.date_range(
                            start=dates['start_date'].iloc[0],
                            end=dates['end_date'].iloc[0]
                        ))
                except Exception:
                    pass
            return pd.DatetimeIndex([])
            
        # Get unique dates from timestamp column
        return pd.DatetimeIndex(self.data['timestamp'].dt.floor('D').unique())
    
    def get_universe(self, date: datetime = None) -> List[str]:
        """Get the universe of available symbols.
        
        Args:
            date: Optional date to filter symbols by availability
            
        Returns:
            List of symbols
        """
        if date is None:
            return sorted(list(self.symbols))
            
        # Filter data by date and return unique symbols
        if self.data is not None:
            date_floor = pd.Timestamp(date).floor('D')
            date_ceil = pd.Timestamp(date).floor('D') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            
            symbols = self.data[
                (self.data['timestamp'] >= date_floor) & 
                (self.data['timestamp'] <= date_ceil)
            ]['symbol'].unique()
            
            return sorted(list(symbols))
            
        return []
    
    def get_available_date_range(self) -> tuple:
        """Get the available date range in the dataset."""
        if self.date_range:
            return self.date_range
            
        # Try to get from database
        if self.use_database and self.db_manager:
            try:
                dates = self.db_manager.get_available_dates()
                if not dates.empty:
                    return (dates['start_date'].iloc[0], dates['end_date'].iloc[0])
            except Exception:
                pass
                
        # Use data if available
        if self.data is not None and not self.data.empty:
            return (self.data['timestamp'].min(), self.data['timestamp'].max())
            
        return (None, None)
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data."""
        start_date, end_date = self.get_available_date_range()
        
        if start_date is None or end_date is None:
            return {
                "row_count": 0,
                "symbol_count": 0,
                "start_date": None,
                "end_date": None,
                "days": 0
            }
            
        # Calculate days
        if pd.api.types.is_datetime64_any_dtype(start_date) and pd.api.types.is_datetime64_any_dtype(end_date):
            days = (end_date - start_date).days + 1
        else:
            days = 0
            
        # Get row and symbol counts
        if self.data is not None:
            row_count = len(self.data)
            symbol_count = len(self.symbols)
        elif self.use_database and self.db_manager:
            try:
                # Get from database
                symbols = self.db_manager.get_available_symbols()
                symbol_count = len(symbols)
                
                # Estimate row count
                row_count = symbol_count * days * 8  # Assuming ~8 data points per day per symbol
            except Exception:
                row_count = 0
                symbol_count = 0
        else:
            row_count = 0
            symbol_count = 0
            
        return {
            "row_count": row_count,
            "symbol_count": symbol_count,
            "start_date": start_date,
            "end_date": end_date,
            "days": days
        }
    
    def ingest_to_database(self, df: pd.DataFrame, source_name: str) -> int:
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
    
    def get_data_for_timestamp(self, timestamp: datetime) -> pd.DataFrame:
        """Get market data for a specific timestamp.
        
        Args:
            timestamp: The timestamp to get data for
            
        Returns:
            DataFrame with market data for the timestamp
        """
        # Try to get data from database first if available
        if self.use_database and self.db_manager:
            try:
                # Use a small window around the timestamp to ensure we get data
                start_time = timestamp - pd.Timedelta(minutes=1)
                end_time = timestamp + pd.Timedelta(minutes=1)
                data = self.db_manager.get_market_data(start_date=start_time, end_date=end_time)
                
                # Filter to get closest timestamps
                if not data.empty:
                    # Get the closest timestamp for each symbol
                    grouped = data.groupby('symbol')
                    closest_data = []
                    
                    for symbol, group in grouped:
                        # Find the row with timestamp closest to the target
                        group['time_diff'] = abs(group['timestamp'] - timestamp)
                        closest_row = group.loc[group['time_diff'].idxmin()].drop('time_diff')
                        closest_data.append(closest_row)
                    
                    if closest_data:
                        return pd.DataFrame(closest_data)
            except Exception as e:
                self.logger.warning(f"Failed to get data from database for timestamp {timestamp}: {e}")
        
        # Fall back to in-memory data if available
        if self.data is not None:
            # Find the closest timestamp for each symbol
            filtered_data = self.data.copy()
            filtered_data['time_diff'] = abs(filtered_data['timestamp'] - timestamp)
            
            # Get the closest timestamp for each symbol
            closest_idx = filtered_data.groupby('symbol')['time_diff'].idxmin()
            result = filtered_data.loc[closest_idx].drop('time_diff', axis=1)
            
            return result
            
        # No data available
        return pd.DataFrame()
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
        """Get all trading days in the specified date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DatetimeIndex with trading days
        """
        # Try to get from database if available
        if self.use_database and self.db_manager:
            try:
                trading_days = self.db_manager.get_trading_days(start_date, end_date)
                if not trading_days.empty:
                    return pd.DatetimeIndex(trading_days['date'])
            except Exception as e:
                self.logger.warning(f"Failed to get trading days from database: {e}")
        
        # Fall back to in-memory data
        if self.data is not None:
            # Filter by date range
            filtered_data = self.data[
                (self.data['timestamp'] >= pd.Timestamp(start_date)) & 
                (self.data['timestamp'] <= pd.Timestamp(end_date))
            ]
            
            # Extract unique dates
            dates = filtered_data['timestamp'].dt.floor('D').unique()
            return pd.DatetimeIndex(dates)
        
        # If no data available, generate business days
        # This is a fallback and may include non-trading days
        return pd.bdate_range(start=start_date, end=end_date)
    
    def get_data_for_date(self, date: datetime) -> pd.DataFrame:
        """Get all market data for a specific date.
        
        Args:
            date: The date to get data for
            
        Returns:
            DataFrame with market data for the date
        """
        date_floor = pd.Timestamp(date).floor('D')
        date_ceil = date_floor + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        
        return self.get_data_for_backtest(date_floor, date_ceil)
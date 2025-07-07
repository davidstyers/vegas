"""Data Layer for the Vegas backtesting engine.

This module provides functionality for ingesting and querying historical market data
using pandas and numpy for efficient vectorized operations.
"""

from typing import Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, date
import os
import pandas as pd
import numpy as np
import glob
import zstandard as zstd
import csv
import io
from pathlib import Path
import logging

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
        """Check if the data layer is initialized with data.
        
        Returns:
            True if data is loaded, False otherwise
        """
        if self.use_database and self.db_manager:
            # Check if there's data in the database
            try:
                dates = self.db_manager.get_available_dates()
                return not dates.empty and dates['day_count'].iloc[0] > 0
            except Exception as e:
                self.logger.warning(f"Database query failed: {e}")
                # Fall back to in-memory check
                return self.data is not None and not self.data.empty
        else:
            return self.data is not None and not self.data.empty
    
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
            if self.use_database and self.db_manager:
                try:
                    # Check if there's data in the database
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
        """Load market data from a single file.
        
        Args:
            file_path: Path to a CSV or compressed CSV file
        """
        self.logger.info(f"Loading data from file: {file_path}")
        
        # Check if it's an OHLCV file
        if file_path.endswith('.ohlcv-1h.csv.zst'):
            if self.use_database and self.db_manager:
                try:
                    self.db_manager.ingest_ohlcv_file(file_path)
                    # Load a sample to initialize the data layer
                    dates = self.db_manager.get_available_dates()
                    if not dates.empty:
                        data = self.db_manager.get_market_data(
                            start_date=dates['start_date'].iloc[0],
                            end_date=dates['start_date'].iloc[0] + pd.Timedelta(days=1)
                        )
                        if not data.empty:
                            self._validate_and_process_data(data)
                            return
                except Exception as e:
                    self.logger.error(f"Failed to ingest OHLCV file into database: {e}")
        
        # Regular file loading
        if file_path.endswith('.zst'):
            # Decompress Zstandard file
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                data_buffer = dctx.decompress(f.read())
                csv_data = io.StringIO(data_buffer.decode('utf-8'))
                df = pd.read_csv(csv_data, parse_dates=['timestamp'])
        else:
            # Regular CSV file
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
        
        # Process the data
        self._validate_and_process_data(df)
        
        # If database is available, also ingest into database
        if self.use_database and self.db_manager:
            try:
                source_name = os.path.basename(file_path)
                self.db_manager.ingest_data(df, source_name)
            except Exception as e:
                self.logger.error(f"Failed to ingest data into database: {e}")
    
    def _load_multiple_files(self, directory: str = None, file_pattern: str = "*.csv*", 
                           max_files: int = None) -> None:
        """Load market data from multiple files.
        
        Args:
            directory: Directory containing data files
            file_pattern: Pattern for matching files
            max_files: Maximum number of files to load
        """
        if directory is None:
            directory = self.data_dir
            
        self.logger.info(f"Loading data from directory: {directory} with pattern: {file_pattern}")
        
        # Check if it's an OHLCV directory
        if file_pattern == "*.ohlcv-1h.csv.zst":
            if self.use_database and self.db_manager:
                try:
                    self.db_manager.ingest_ohlcv_directory(directory, file_pattern, max_files)
                    # Load a sample to initialize the data layer
                    dates = self.db_manager.get_available_dates()
                    if not dates.empty:
                        data = self.db_manager.get_market_data(
                            start_date=dates['start_date'].iloc[0],
                            end_date=dates['start_date'].iloc[0] + pd.Timedelta(days=1)
                        )
                        if not data.empty:
                            self._validate_and_process_data(data)
                            return
                except Exception as e:
                    self.logger.error(f"Failed to ingest OHLCV files into database: {e}")
        
        # Find data files
        search_path = os.path.join(directory, file_pattern)
        files = sorted(glob.glob(search_path))
        
        if not files:
            # Try subdirectories if no files found
            for subdir in os.listdir(directory):
                subdir_path = os.path.join(directory, subdir)
                if os.path.isdir(subdir_path):
                    search_path = os.path.join(subdir_path, file_pattern)
                    subdir_files = sorted(glob.glob(search_path))
                    files.extend(subdir_files)
        
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
                if file.endswith('.zst'):
                    # Decompress Zstandard file
                    with open(file, 'rb') as f:
                        dctx = zstd.ZstdDecompressor()
                        data_buffer = dctx.decompress(f.read())
                        csv_data = io.StringIO(data_buffer.decode('utf-8'))
                        df = pd.read_csv(csv_data, parse_dates=['timestamp'])
                else:
                    # Regular CSV file
                    df = pd.read_csv(file, parse_dates=['timestamp'])
                
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
    
    def _validate_and_process_data(self, df: pd.DataFrame) -> None:
        """Validate and process loaded data.
        
        Args:
            df: DataFrame with market data
        """
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Make sure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp and symbol
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # Store the data
        self.data = df
        
        # Update symbols and date range
        self.symbols = set(df['symbol'].unique())
        self.date_range = (df['timestamp'].min(), df['timestamp'].max())
        
        self.logger.info(f"Loaded {len(df)} data points for {len(self.symbols)} symbols "
                        f"from {self.date_range[0].date()} to {self.date_range[1].date()}")
    
    def get_data_for_backtest(self, start: datetime, end: datetime, symbols: List[str] = None) -> pd.DataFrame:
        """Get market data for backtesting in a vectorized format.
        
        Args:
            start: Start date for the backtest
            end: End date for the backtest
            symbols: Optional list of symbols to include (if None, all symbols are included)
            
        Returns:
            DataFrame with market data filtered by date range and symbols
        """
        # Try to use database if available
        if self.use_database and self.db_manager:
            try:
                data = self.db_manager.get_market_data(start, end, symbols)
                
                if not data.empty:
                    self.logger.info(f"Retrieved {len(data)} rows from database for backtest")
                    return data
                else:
                    self.logger.warning("No data found in database, falling back to in-memory data")
            except Exception as e:
                self.logger.error(f"Database query failed, falling back to in-memory data: {e}")
        
        # Fall back to in-memory data
        if not self.is_initialized():
            raise ValueError("No data is loaded. Load data first.")
            
        # Filter by date range
        mask = (self.data['timestamp'] >= start) & (self.data['timestamp'] <= end)
        data = self.data[mask]
        
        # Filter by symbols if specified
        if symbols:
            data = data[data['symbol'].isin(symbols)]
            
        return data
    
    def get_unique_dates(self) -> pd.DatetimeIndex:
        """Get unique dates in the dataset.
        
        Returns:
            DatetimeIndex of unique dates
        """
        if self.use_database and self.db_manager:
            try:
                dates_df = self.db_manager.query_to_df("SELECT DISTINCT DATE_TRUNC('day', timestamp) as date FROM market_data ORDER BY date")
                if not dates_df.empty:
                    return pd.DatetimeIndex(dates_df['date'].values)
            except Exception as e:
                self.logger.error(f"Database query failed, falling back to in-memory data: {e}")
        
        # Fall back to in-memory data
        if not self.is_initialized():
            raise ValueError("No data is loaded. Load data first.")
            
        return pd.DatetimeIndex(self.data['timestamp'].dt.floor('D').unique()).sort_values()
    
    def get_universe(self, date: datetime = None) -> List[str]:
        """Get the universe of available symbols.
        
        Args:
            date: Optional date to get symbols available on that date
            
        Returns:
            List of symbols
        """
        if self.use_database and self.db_manager:
            try:
                if date is None:
                    symbols_df = self.db_manager.query_to_df("SELECT DISTINCT symbol FROM market_data ORDER BY symbol")
                else:
                    date_str = date.strftime("%Y-%m-%d")
                    symbols_df = self.db_manager.query_to_df(
                        f"SELECT DISTINCT symbol FROM market_data WHERE DATE_TRUNC('day', timestamp) = '{date_str}' ORDER BY symbol"
                    )
                
                if not symbols_df.empty:
                    return symbols_df['symbol'].tolist()
            except Exception as e:
                self.logger.error(f"Database query failed, falling back to in-memory data: {e}")
        
        # Fall back to in-memory data
        if not self.is_initialized():
            raise ValueError("No data is loaded. Load data first.")
            
        if date is None:
            return sorted(list(self.symbols))
        else:
            # Get symbols available on the given date
            date_start = pd.Timestamp(date.year, date.month, date.day)
            date_end = date_start + pd.Timedelta(days=1)
            
            symbols = self.data[(self.data['timestamp'] >= date_start) & 
                              (self.data['timestamp'] < date_end)]['symbol'].unique()
            return sorted(list(symbols))
    
    def get_available_date_range(self) -> Tuple[datetime, datetime]:
        """Get the available date range in the dataset.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        if self.use_database and self.db_manager:
            try:
                dates_df = self.db_manager.get_available_dates()
                if not dates_df.empty:
                    return (dates_df['start_date'].iloc[0], dates_df['end_date'].iloc[0])
            except Exception as e:
                self.logger.error(f"Database query failed, falling back to in-memory data: {e}")
        
        # Fall back to in-memory data
        if not self.is_initialized():
            raise ValueError("No data is loaded. Load data first.")
            
        return self.date_range
    
    def get_data_info(self) -> Dict[str, any]:
        """Get information about the loaded data.
        
        Returns:
            Dictionary with information about the data
        """
        if self.use_database and self.db_manager:
            try:
                dates_df = self.db_manager.get_available_dates()
                symbols_df = self.db_manager.get_available_symbols()
                
                if not dates_df.empty and not symbols_df.empty:
                    return {
                        "row_count": symbols_df['record_count'].sum(),
                        "symbol_count": len(symbols_df),
                        "start_date": dates_df['start_date'].iloc[0],
                        "end_date": dates_df['end_date'].iloc[0],
                        "day_count": dates_df['day_count'].iloc[0],
                        "database_size_mb": round(self.db_manager.get_database_size() / (1024 * 1024), 2)
                    }
            except Exception as e:
                self.logger.error(f"Database query failed, falling back to in-memory data: {e}")
        
        # Fall back to in-memory data
        if not self.is_initialized():
            return {
                "row_count": 0,
                "symbol_count": 0,
                "start_date": None,
                "end_date": None,
                "day_count": 0,
                "database_size_mb": 0
            }
        
        return {
            "row_count": len(self.data),
            "symbol_count": len(self.symbols),
            "start_date": self.date_range[0],
            "end_date": self.date_range[1],
            "day_count": len(self.data['timestamp'].dt.floor('D').unique()),
            "database_size_mb": 0
        }
    
    def ingest_to_database(self, df: pd.DataFrame, source_name: str) -> int:
        """Ingest a DataFrame directly into the database.
        
        Args:
            df: DataFrame with market data
            source_name: Name of the data source
            
        Returns:
            Number of rows ingested
        """
        if not self.use_database or not self.db_manager:
            raise ValueError("Database is not available")
            
        return self.db_manager.ingest_data(df, source_name)
    
    def ingest_ohlcv_file(self, file_path: str) -> int:
        """Ingest an OHLCV file into the database.
        
        Args:
            file_path: Path to the OHLCV file
            
        Returns:
            Number of rows ingested
        """
        if not self.use_database or not self.db_manager:
            raise ValueError("Database is not available")
            
        return self.db_manager.ingest_ohlcv_file(file_path)
    
    def ingest_ohlcv_directory(self, directory: str, max_files: int = None) -> int:
        """Ingest all OHLCV files in a directory.
        
        Args:
            directory: Directory containing OHLCV files
            max_files: Maximum number of files to ingest
            
        Returns:
            Number of rows ingested
        """
        if not self.use_database or not self.db_manager:
            raise ValueError("Database is not available")
            
        return self.db_manager.ingest_ohlcv_directory(directory, max_files=max_files)
    
    def close(self) -> None:
        """Close all connections and clean up resources."""
        if self.use_database and self.db_manager:
            self.db_manager.close()
            self.logger.info("Database connections closed")
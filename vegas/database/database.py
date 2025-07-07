"""Database management layer for the Vegas backtesting engine.

This module provides functionality for managing market data using DuckDB and Parquet files.
"""

import os
import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set
from datetime import datetime, date
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


class ParquetManager:
    """Manager for Parquet file operations.
    
    This class handles converting market data to Parquet format and writing/reading files.
    """
    
    def __init__(self, data_dir: str = "db"):
        """Initialize the ParquetManager.
        
        Args:
            data_dir: Directory for storing Parquet files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger("vegas.database.parquet")
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create partitioned directory if it doesn't exist
        os.makedirs(os.path.join(self.data_dir, "partitioned"), exist_ok=True)
    
    def write_dataframe_to_parquet(self, df: pd.DataFrame, file_path: str) -> str:
        """Write a DataFrame to a Parquet file.
        
        Args:
            df: DataFrame to write
            file_path: Path to write the Parquet file
            
        Returns:
            Path to the written file
        """
        # Make sure the directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Convert to PyArrow Table and write to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression="snappy")
        
        self.logger.info(f"Wrote {len(df)} rows to {file_path}")
        return file_path
    
    def write_data_partitioned(self, df: pd.DataFrame, partition_cols: List[str]) -> List[str]:
        """Write data partitioned by specified columns.
        
        Args:
            df: DataFrame to write
            partition_cols: Columns to partition by
            
        Returns:
            List of paths to written files
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided, nothing to write")
            return []
        
        # Ensure all partition columns exist
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Partition columns not found in DataFrame: {missing_cols}")
        
        # Create partitioned dataset
        table = pa.Table.from_pandas(df)
        
        # Define the partition path
        partition_path = os.path.join(self.data_dir, "partitioned")
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table, 
            partition_path,
            partition_cols=partition_cols,
            compression="snappy"
        )
        
        # Find all written files
        parquet_files = glob.glob(os.path.join(partition_path, "**", "*.parquet"), recursive=True)
        
        self.logger.info(f"Wrote {len(df)} rows to {len(parquet_files)} partition files")
        return parquet_files
    
    def read_parquet_file(self, file_path: str) -> pd.DataFrame:
        """Read a Parquet file into a DataFrame.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            DataFrame with the file contents
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        self.logger.info(f"Read {len(df)} rows from {file_path}")
        return df
    
    def read_partitioned_dataset(self, base_dir: str = None, filters=None) -> pd.DataFrame:
        """Read a partitioned dataset with optional filters.
        
        Args:
            base_dir: Base directory of the partitioned dataset
            filters: PyArrow filters to apply
            
        Returns:
            DataFrame with the combined dataset
        """
        if base_dir is None:
            base_dir = os.path.join(self.data_dir, "partitioned")
            
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")
        
        # Check if there are any parquet files
        parquet_files = glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            self.logger.warning(f"No Parquet files found in {base_dir}")
            return pd.DataFrame()
        
        dataset = pq.ParquetDataset(base_dir, filters=filters)
        table = dataset.read()
        df = table.to_pandas()
        
        self.logger.info(f"Read {len(df)} rows from partitioned dataset in {base_dir}")
        return df


class DatabaseManager:
    """Manager for DuckDB operations.
    
    This class provides an interface to the DuckDB database for managing market data.
    """
    
    def __init__(self, db_path: str = "db/vegas.duckdb", parquet_dir: str = "db"):
        """Initialize the DatabaseManager.
        
        Args:
            db_path: Path to the DuckDB database file
            parquet_dir: Directory for storing Parquet files
        """
        self.db_path = db_path
        self.logger = logging.getLogger("vegas.database")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize ParquetManager
        self.parquet_manager = ParquetManager(parquet_dir)
        
        # Initialize the connection
        self._conn = None
        self.connect()
    
    def connect(self) -> None:
        """Connect to the DuckDB database."""
        try:
            self._conn = duckdb.connect(self.db_path)
            self.logger.info(f"Connected to DuckDB database at {self.db_path}")
            
            # Enable automatic loading of Parquet files
            self._conn.execute("INSTALL parquet; LOAD parquet;")
            
            # Initialize database schema if needed
            self._initialize_schema()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema."""
        # Create symbols table
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            exchange VARCHAR,
            asset_type VARCHAR,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create data sources table
        self._conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS data_sources_id_seq;
        """)
        
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY DEFAULT nextval('data_sources_id_seq'),
            source_name VARCHAR NOT NULL,
            source_path VARCHAR NOT NULL,
            format VARCHAR NOT NULL,
            row_count INTEGER,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create a view for market data - check if partitioned directory exists and has files
        partitioned_dir = os.path.abspath(os.path.join(os.path.dirname(self.db_path), "partitioned"))
        parquet_files = glob.glob(os.path.join(partitioned_dir, "**", "*.parquet"), recursive=True)
        
        if parquet_files:
            try:
                # Create the view using the entire partitioned directory with a wildcard
                self._conn.execute(f"""
                CREATE OR REPLACE VIEW market_data AS
                SELECT * FROM parquet_scan('{partitioned_dir}/**/*.parquet')
                """)
                self.logger.info(f"Created market_data view using {partitioned_dir}/**/*.parquet")
            except Exception as e:
                self.logger.warning(f"Failed to create market_data view: {e}")
                self._create_empty_market_data_view()
        else:
            self.logger.info("No Parquet files found, creating empty market_data view")
            self._create_empty_market_data_view()
        
        self.logger.info("Database schema initialized")
    
    def _create_empty_market_data_view(self) -> None:
        """Create an empty market_data view."""
        self._conn.execute("""
        CREATE OR REPLACE VIEW market_data AS
        SELECT 
            CAST(NULL AS TIMESTAMP) as timestamp,
            CAST(NULL AS VARCHAR) as symbol,
            CAST(NULL AS DOUBLE) as open,
            CAST(NULL AS DOUBLE) as high,
            CAST(NULL AS DOUBLE) as low,
            CAST(NULL AS DOUBLE) as close,
            CAST(NULL AS BIGINT) as volume,
            CAST(NULL AS INTEGER) as year,
            CAST(NULL AS INTEGER) as month
        WHERE 1=0
        """)
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, parameters: tuple = ()) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query on the DuckDB connection.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            
        Returns:
            DuckDB query result
        """
        if not self._conn:
            self.connect()
            
        try:
            result = self._conn.execute(query, parameters)
            return result
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            raise
    
    def query_to_df(self, query: str, parameters: tuple = ()) -> pd.DataFrame:
        """Execute a query and return the result as a DataFrame.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            
        Returns:
            DataFrame with query results
        """
        result = self.execute_query(query, parameters)
        return result.df()
    
    def ingest_data(self, df: pd.DataFrame, source_name: str) -> int:
        """Ingest data into the database via Parquet files.
        
        Args:
            df: DataFrame with market data
            source_name: Name of the data source
            
        Returns:
            Number of rows ingested
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided, nothing to ingest")
            return 0
        
        # Check if this source has already been ingested
        try:
            existing_sources = self.query_to_df("SELECT * FROM data_sources WHERE source_name = ?", (source_name,))
            if not existing_sources.empty:
                self.logger.warning(f"Data source '{source_name}' has already been ingested. Skipping to prevent duplicate data.")
                return 0
        except Exception as e:
            # If the data_sources table doesn't exist yet, this is the first run after deleting the DB
            self.logger.debug(f"Error checking for existing sources: {e}")
            # Continue with ingestion
        
        # Validate required columns for market data
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data: {df.columns}")
        
        # Make sure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create year and month columns for partitioning
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        
        # Write the data to Parquet files, partitioned by year and month
        # This reduces the number of partitions compared to year/date/symbol
        parquet_files = self.parquet_manager.write_data_partitioned(df, partition_cols=['year', 'month'])
        
        # Update the symbols table
        symbols = df['symbol'].unique().tolist()
        if symbols:
            values = ", ".join(f"('{symbol}')" for symbol in symbols)
            
            self._conn.execute(f"""
            INSERT OR IGNORE INTO symbols (symbol)
            VALUES {values}
            """)
        
        # Add an entry to the data sources table
        self._conn.execute("""
        INSERT INTO data_sources (source_name, source_path, format, row_count, start_date, end_date)
        VALUES (?, 'db/partitioned', 'parquet', ?, ?, ?)
        """, (source_name, len(df), df['timestamp'].min(), df['timestamp'].max()))
        
        # Try to refresh the market data view using absolute paths
        try:
            # Get the absolute path to the partitioned directory
            partitioned_dir = os.path.abspath(os.path.join(os.path.dirname(self.db_path), "partitioned"))
            
            # Create the view using the entire partitioned directory with a wildcard
            self._conn.execute(f"""
            CREATE OR REPLACE VIEW market_data AS
            SELECT * FROM parquet_scan('{partitioned_dir}/**/*.parquet')
            """)
            self.logger.info(f"Created market_data view using {partitioned_dir}/**/*.parquet")
        except Exception as e:
            self.logger.warning(f"Failed to refresh market_data view: {e}")
            # Create an empty view if the refresh fails
            self._create_empty_market_data_view()
        
        self.logger.info(f"Ingested {len(df)} rows from {source_name}")
        return len(df)
    
    def ingest_ohlcv_file(self, file_path: str) -> int:
        """Ingest an OHLCV file into the database.
        
        Args:
            file_path: Path to the OHLCV file
            
        Returns:
            Number of rows ingested
        """
        import zstandard as zstd
        import io
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file is an OHLCV file
        if not file_path.endswith('.ohlcv-1h.csv.zst'):
            raise ValueError(f"Not an OHLCV file: {file_path}")
        
        # Extract date from filename
        match = re.search(r'(\d{8})\.ohlcv-1h\.csv\.zst$', file_path)
        if not match:
            raise ValueError(f"Could not extract date from filename: {file_path}")
        
        date_str = match.group(1)
        source_name = os.path.basename(file_path)
        
        # Check if this file has already been ingested
        try:
            existing_sources = self.query_to_df("SELECT * FROM data_sources WHERE source_name = ?", (source_name,))
            if not existing_sources.empty:
                self.logger.warning(f"File {source_name} has already been ingested. Skipping to prevent duplicate data.")
                return 0
        except Exception as e:
            # If the data_sources table doesn't exist yet, this is the first run after deleting the DB
            self.logger.debug(f"Error checking for existing sources: {e}")
            # Continue with ingestion
        
        # Decompress and read the file
        self.logger.info(f"Reading OHLCV file: {file_path}")
        
        # Use subprocess to call zstdcat directly
        import subprocess
        import tempfile
        
        self.logger.info(f"Using zstdcat to decompress {file_path}")
        try:
            # Create a temporary file to store the decompressed data
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use zstdcat to decompress the file
            cmd = f"zstdcat {file_path} > {temp_path}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Read the decompressed file
            self.logger.info(f"Reading decompressed file from {temp_path}")
            df = pd.read_csv(temp_path)
            self.logger.info(f"Read {len(df)} rows from {temp_path}")
            
            # Clean up
            os.unlink(temp_path)
            
            # Process the data
            self.logger.info(f"Processing {len(df)} rows from {file_path}")
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'ts_event': 'timestamp',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'open': 'open',
                'volume': 'volume',
                'symbol': 'symbol'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ingest the data
            return self.ingest_data(df, source_name)
        except Exception as e:
            self.logger.error(f"Error processing OHLCV file {file_path}: {e}")
            raise ValueError(f"Failed to process OHLCV file {file_path}: {e}")
    
    def ingest_ohlcv_directory(self, directory: str = "data", pattern: str = "*.ohlcv-1h.csv.zst", max_files: int = None) -> int:
        """Ingest all OHLCV files in a directory.
        
        Args:
            directory: Directory containing OHLCV files
            pattern: Pattern for matching OHLCV files
            max_files: Maximum number of files to ingest
            
        Returns:
            Number of rows ingested
        """
        # Find all OHLCV files
        search_path = os.path.join(directory, pattern)
        files = sorted(glob.glob(search_path))
        
        if not files:
            raise FileNotFoundError(f"No OHLCV files found in {directory} with pattern {pattern}")
        
        # Limit the number of files if specified
        if max_files:
            files = files[:max_files]
        
        self.logger.info(f"Found {len(files)} OHLCV files to ingest")
        
        # Ingest each file
        total_rows = 0
        skipped_files = 0
        
        for file in files:
            try:
                rows = self.ingest_ohlcv_file(file)
                if rows > 0:
                    total_rows += rows
                    self.logger.info(f"Ingested {rows} rows from {file}")
                else:
                    skipped_files += 1
            except Exception as e:
                self.logger.error(f"Error ingesting file {file}: {e}")
        
        if skipped_files > 0:
            self.logger.info(f"Skipped {skipped_files} files that were already ingested")
            
        return total_rows
    
    def get_available_dates(self) -> pd.DataFrame:
        """Get the range of available dates in the database.
        
        Returns:
            DataFrame with date statistics
        """
        try:
            return self.query_to_df("""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as day_count
            FROM market_data
            """)
        except Exception as e:
            self.logger.error(f"Error getting available dates: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame({
                'start_date': [None],
                'end_date': [None],
                'day_count': [0]
            })
    
    def get_available_symbols(self) -> pd.DataFrame:
        """Get the list of available symbols in the database.
        
        Returns:
            DataFrame with symbol statistics
        """
        try:
            return self.query_to_df("""
            SELECT 
                symbol,
                COUNT(*) as record_count,
                MIN(timestamp) as first_date,
                MAX(timestamp) as last_date
            FROM market_data
            GROUP BY symbol
            ORDER BY symbol
            """)
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return pd.DataFrame()
    
    def get_market_data(self, start_date: datetime = None, end_date: datetime = None, 
                       symbols: List[str] = None) -> pd.DataFrame:
        """Query market data with optional filters.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            symbols: Optional list of symbols to include
            
        Returns:
            DataFrame with market data
        """
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        if symbols and len(symbols) > 0:
            placeholders = ", ".join(["?"] * len(symbols))
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
            
        query += " ORDER BY timestamp, symbol"
        
        return self.query_to_df(query, tuple(params))
    
    def get_data_sources(self) -> pd.DataFrame:
        """Get information about ingested data sources.
        
        Returns:
            DataFrame with data source information
        """
        return self.query_to_df("SELECT * FROM data_sources ORDER BY added_date DESC")
    
    def get_database_size(self) -> int:
        """Get the size of the database file in bytes.
        
        Returns:
            Size of database file in bytes
        """
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0 
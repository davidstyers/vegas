"""Database management layer for the Vegas backtesting engine.

This module provides functionality for managing market data using DuckDB and Parquet files.
"""

import os
import glob
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


class ParquetManager:
    """Manager for Parquet file operations."""
    
    def __init__(self, data_dir: str = "db"):
        """Initialize the ParquetManager.
        
        Args:
            data_dir: Directory for storing Parquet files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger("vegas.database.parquet")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
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
        
        # Define the partition path and write partitioned dataset
        partition_path = os.path.join(self.data_dir, "partitioned")
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, partition_path, partition_cols=partition_cols, compression="snappy")
        
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
    """Manager for DuckDB operations."""
    
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
            
            # Initialize database schema
            self._initialize_schema()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema."""
        try:
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
            self._conn.execute("CREATE SEQUENCE IF NOT EXISTS data_sources_id_seq;")
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
            
            # Create market_data view
            self._create_market_data_view()
            
            self.logger.info("Database schema initialized")
        except Exception as e:
            self.logger.error(f"Error initializing schema: {e}")
            raise
    
    def _create_market_data_view(self) -> None:
        """Create or update the market_data view."""
        # Check if partitioned directory exists and has files
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
    
    def _create_empty_market_data_view(self) -> None:
        """Create an empty market_data view."""
        self._conn.execute("""
        CREATE OR REPLACE VIEW market_data AS
        SELECT 
            CAST(NULL AS TIMESTAMP) as timestamp,
            CAST(NULL AS VARCHAR) as symbol,
            CAST(NULL AS FLOAT) as open,
            CAST(NULL AS FLOAT) as high,
            CAST(NULL AS FLOAT) as low,
            CAST(NULL AS FLOAT) as close,
            CAST(NULL AS INTEGER) as volume
        WHERE FALSE
        """)
        self.logger.info("Created empty market_data view")
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
                self._conn = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
    
    def execute_query(self, query: str, parameters: tuple = ()) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query and return the result relation.
        
        Args:
            query: SQL query to execute
            parameters: Optional tuple of parameters for the query
            
        Returns:
            DuckDB relation object with query results
        """
        if not self._conn:
            self.connect()
            
        try:
            return self._conn.execute(query, parameters)
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}\nQuery: {query}")
            raise
    
    def query_to_df(self, query: str, parameters: tuple = ()) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.
        
        Args:
            query: SQL query to execute
            parameters: Optional tuple of parameters for the query
            
        Returns:
            DataFrame with query results
        """
        try:
            result = self.execute_query(query, parameters)
            return result.fetchdf()
        except Exception as e:
            self.logger.error(f"Failed to convert query results to DataFrame: {e}")
            return pd.DataFrame()
    
    def ingest_data(self, df: pd.DataFrame, source_name: str) -> int:
        """Ingest market data into the database.
        
        Args:
            df: DataFrame with market data
            source_name: Name or identifier for the data source
            
        Returns:
            Number of rows ingested
        """
        if df.empty:
            self.logger.warning(f"Empty DataFrame provided for source {source_name}, nothing to ingest")
            return 0
        
        required_columns = ['timestamp', 'symbol']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data for source {source_name}")
        
        try:
            # Ensure timestamp is a datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Add year and month columns for partitioning
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            
            # Add partition path
            source_path = f"ingested/{source_name.replace('.', '_')}"
            
            # Write partitioned data
            partition_cols = ['year', 'month', 'symbol']
            self.parquet_manager.write_data_partitioned(df, partition_cols)
            
            # Record the data source
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            
            # Insert or update data source record
            self._conn.execute("""
            INSERT INTO data_sources (source_name, source_path, format, row_count, start_date, end_date)
            VALUES (?, ?, 'parquet', ?, ?, ?)
            """, (source_name, source_path, len(df), min_date, max_date))
            
            # Insert symbols if they don't exist
            symbols = df['symbol'].unique()
            for symbol in symbols:
                self._conn.execute("""
                INSERT INTO symbols (symbol) VALUES (?)
                ON CONFLICT (symbol) DO NOTHING
                """, (symbol,))
            
            # Refresh the market_data view to include the new data
            self._create_market_data_view()
            
            self.logger.info(f"Ingested {len(df)} rows from {source_name}")
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Data ingestion failed for {source_name}: {e}")
            raise
    
    def ingest_ohlcv_file(self, file_path: str) -> int:
        """Ingest an OHLCV file into the database.
        
        Args:
            file_path: Path to OHLCV file (.ohlcv-1h.csv.zst format)
            
        Returns:
            Number of rows ingested
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OHLCV file not found: {file_path}")
            
        if not file_path.endswith('.ohlcv-1h.csv.zst'):
            raise ValueError(f"Invalid OHLCV file format for {file_path}. Expected .ohlcv-1h.csv.zst")
            
        self.logger.info(f"Ingesting OHLCV file: {file_path}")
        
        try:
            # Decompress the file
            import zstandard as zstd
            import io
            import csv
            
            with open(file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                data_buffer = dctx.decompress(f.read())
                csv_data = io.StringIO(data_buffer.decode('utf-8'))
                
                # Load CSV data
                df = pd.read_csv(csv_data)
                
                # Validate required columns
                required_columns = ['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns in OHLCV file: {missing_columns}")
                
                # Rename ts_event to timestamp for consistency
                df = df.rename(columns={'ts_event': 'timestamp'})
                
                # Ensure timestamp is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Get source name from filename
                source_name = os.path.basename(file_path)
                
                # Ingest the data
                return self.ingest_data(df, source_name)
                
        except Exception as e:
            self.logger.error(f"Failed to ingest OHLCV file {file_path}: {e}")
            raise
    
    def ingest_ohlcv_directory(self, directory: str = "data", pattern: str = "*.ohlcv-1h.csv.zst", max_files: int = None) -> int:
        """Ingest all OHLCV files in a directory.
        
        Args:
            directory: Directory containing OHLCV files
            pattern: Glob pattern for matching files
            max_files: Maximum number of files to ingest
            
        Returns:
            Total number of rows ingested
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        # Find all matching files
        search_path = os.path.join(directory, pattern)
        files = sorted(glob.glob(search_path, recursive=True))
        
        if not files:
            # Check subdirectories if no files found
            for root, _, _ in os.walk(directory):
                search_path = os.path.join(root, pattern)
                subdir_files = sorted(glob.glob(search_path))
                files.extend(subdir_files)
        
        if not files:
            self.logger.warning(f"No OHLCV files found in {directory} with pattern {pattern}")
            return 0
            
        # Limit the number of files if specified
        if max_files:
            files = files[:max_files]
            
        self.logger.info(f"Found {len(files)} OHLCV files to ingest")
        
        # Ingest each file
        total_rows = 0
        success_count = 0
        for file_path in files:
            try:
                rows = self.ingest_ohlcv_file(file_path)
                total_rows += rows
                success_count += 1
                self.logger.info(f"Successfully ingested {file_path}: {rows} rows")
            except Exception as e:
                self.logger.error(f"Failed to ingest {file_path}: {e}")
        
        self.logger.info(f"Ingested {total_rows} rows from {success_count}/{len(files)} OHLCV files")
        return total_rows
    
    def get_available_dates(self) -> pd.DataFrame:
        """Get the date range and day count available in the database.
        
        Returns:
            DataFrame with start_date, end_date, and day_count
        """
        try:
            result = self.query_to_df("""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as day_count
            FROM market_data
            """)
            
            # Handle case where no data is available
            if result.empty or pd.isna(result['start_date'].iloc[0]):
                return pd.DataFrame({'start_date': [None], 'end_date': [None], 'day_count': [0]})
                
            return result
        except Exception as e:
            self.logger.error(f"Failed to get available dates: {e}")
            return pd.DataFrame({'start_date': [None], 'end_date': [None], 'day_count': [0]})
    
    def get_available_symbols(self) -> pd.DataFrame:
        """Get the symbols available in the database.
        
        Returns:
            DataFrame with symbols and their record counts
        """
        try:
            return self.query_to_df("""
            SELECT 
                symbol,
                COUNT(*) as record_count
            FROM market_data
            GROUP BY symbol
            ORDER BY symbol
            """)
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return pd.DataFrame(columns=['symbol', 'record_count'])
    
    def get_market_data(self, start_date: datetime = None, end_date: datetime = None, 
                      symbols: List[str] = None) -> pd.DataFrame:
        """Query market data from the database.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            symbols: Optional list of symbols to filter by
            
        Returns:
            DataFrame with market data
        """
        try:
            query = "SELECT * FROM market_data"
            conditions = []
            params = []
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
                
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
                
            if symbols and len(symbols) > 0:
                # For better performance with many symbols, use IN clause
                if len(symbols) <= 10:
                    symbols_list = ", ".join([f"'{s}'" for s in symbols])
                    conditions.append(f"symbol IN ({symbols_list})")
                else:
                    # For many symbols, create a temporary table and join
                    self._conn.execute("CREATE TEMP TABLE IF NOT EXISTS temp_symbols (symbol VARCHAR)")
                    self._conn.execute("DELETE FROM temp_symbols")
                    for symbol in symbols:
                        self._conn.execute("INSERT INTO temp_symbols VALUES (?)", (symbol,))
                    conditions.append("symbol IN (SELECT symbol FROM temp_symbols)")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += " ORDER BY timestamp, symbol"
            
            return self.query_to_df(query, tuple(params))
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def get_data_sources(self) -> pd.DataFrame:
        """Get information about data sources in the database.
        
        Returns:
            DataFrame with data source information
        """
        return self.query_to_df("SELECT * FROM data_sources ORDER BY added_date DESC")
    
    def get_database_size(self) -> int:
        """Get the size of the database file in bytes.
        
        Returns:
            Database file size in bytes
        """
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0 
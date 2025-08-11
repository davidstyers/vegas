"""Database management layer for the Vegas backtesting engine.

This module provides functionality for managing market data using DuckDB and Parquet files.
Optimized to use polars for high-performance data operations.
"""

import glob
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import List

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetManager:
    """Utility for Parquet file I/O and partitioned dataset management.

    Ensures directories exist, writes Polars DataFrames to Parquet using
    sensible defaults, and can materialize partitioned datasets while guarding
    against excessive partition counts.
    """

    def __init__(self, data_dir: str = "db"):
        """Initialize the manager with a base data directory.

        :param data_dir: Directory used for reading/writing Parquet files.
        :type data_dir: str
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger("vegas.database.parquet")
        # Use the logger configured by CLI - don't add handlers

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "partitioned"), exist_ok=True)

    def write_dataframe_to_parquet(self, df: pl.DataFrame, file_path: str) -> str:
        """Write a Polars `DataFrame` to a Parquet file and return the path.

        :param df: DataFrame to write.
        :type df: pl.DataFrame
        :param file_path: Destination file path.
        :type file_path: str
        :returns: Path to the written file.
        :rtype: str
        :raises Exception: Propagates write errors.
        """
        # Make sure the directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Write polars DataFrame to Parquet
        df.write_parquet(file_path, compression="snappy")

        self.logger.info(f"Wrote {len(df)} rows to {file_path}")
        return file_path

    def write_data_partitioned(
        self, df: pl.DataFrame, partition_cols: List[str]
    ) -> List[str]:
        """Write a dataset partitioned by specific columns and return file paths.

        :param df: DataFrame to write.
        :type df: pl.DataFrame
        :param partition_cols: Columns used to partition the dataset.
        :type partition_cols: list[str]
        :returns: List of written Parquet file paths.
        :rtype: list[str]
        :raises ValueError: If any partition columns are missing.
        """
        if df.is_empty():
            self.logger.warning("Empty DataFrame provided, nothing to write")
            return []

        # Ensure all partition columns exist
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Partition columns not found in DataFrame: {missing_cols}"
            )

        # Check number of potential partitions to prevent exceeding limits
        # PyArrow has a limit of 1024 partitions by default
        MAX_PARTITIONS = 1000  # Setting slightly below the limit to be safe

        # Calculate potential partitions
        potential_partitions = 1
        for col in partition_cols:
            unique_values = df.select(pl.col(col).n_unique()).item()
            potential_partitions *= unique_values

        if potential_partitions > MAX_PARTITIONS:
            self.logger.warning(
                f"Potential partitions ({potential_partitions}) exceeds the maximum recommended ({MAX_PARTITIONS}). "
                f"Consider reducing the granularity of your partition columns or using fewer partition columns."
            )

            # Simplified partitioning to prevent errors
            reduced_cols = partition_cols.copy()
            while potential_partitions > MAX_PARTITIONS and len(reduced_cols) > 1:
                removed_col = reduced_cols.pop()
                potential_partitions //= (
                    df.select(pl.col(removed_col).n_unique()).item() or 1
                )
                self.logger.warning(
                    f"Removed '{removed_col}' from partition columns to reduce partitions to {potential_partitions}"
                )

            partition_cols = reduced_cols
            self.logger.info(f"Using reduced partition columns: {partition_cols}")

        # Define the partition path and write partitioned dataset
        partition_path = os.path.join(self.data_dir, "partitioned")

        # Make a copy of the dataframe to preserve partition columns in the data
        # This helps prevent schema mismatches when partition schemes change
        write_df = df.copy()

        # Ensure schema consistency by keeping partition columns in the data
        # Note that PyArrow will still use these for partitioning but also keep them in the data
        table = pa.Table.from_pandas(write_df)
        pq.write_to_dataset(
            table,
            partition_path,
            partition_cols=partition_cols,
            compression="snappy",
            # Explicitly tell PyArrow to preserve the partition columns in the written data
            existing_data_behavior="overwrite_or_ignore",
        )

        # Find all written files
        parquet_files = glob.glob(
            os.path.join(partition_path, "**", "*.parquet"), recursive=True
        )

        self.logger.info(
            f"Wrote {len(df)} rows to {len(parquet_files)} partition files"
        )
        return parquet_files

    def read_parquet_file(self, file_path: str) -> pl.DataFrame:
        """Read a Parquet file into a Polars DataFrame.

        :param file_path: File path to read.
        :type file_path: str
        :returns: DataFrame containing the file contents.
        :rtype: pl.DataFrame
        :raises FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pl.read_parquet(file_path)

        self.logger.info(f"Read {len(df)} rows from {file_path}")
        return df

    def read_partitioned_dataset(
        self, base_dir: str = None, filters=None
    ) -> pl.DataFrame:
        """Read a partitioned dataset and return a combined DataFrame.

        :param base_dir: Base directory; defaults to `<data_dir>/partitioned`.
        :type base_dir: Optional[str]
        :param filters: Optional PyArrow dataset filters.
        :type filters: Any
        :returns: Combined DataFrame across partitions or empty if none found.
        :rtype: pl.DataFrame
        :raises FileNotFoundError: If the directory does not exist.
        """
        if base_dir is None:
            base_dir = os.path.join(self.data_dir, "partitioned")

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

        # Check if there are any parquet files
        parquet_files = glob.glob(
            os.path.join(base_dir, "**", "*.parquet"), recursive=True
        )
        if not parquet_files:
            self.logger.warning(f"No Parquet files found in {base_dir}")
            return pl.DataFrame()

        try:
            # Use polars to read the partitioned dataset
            df = pl.read_parquet(base_dir, use_pyarrow_dataset=True)

            # Process partition columns - extract from path if they don't exist in the data
            for col in ["year", "month"]:
                if col not in df.columns and f"{col}=" in "\n".join(parquet_files):
                    self.logger.info(f"Extracting '{col}' column from partition paths")
                    # Extract values from paths using regex
                    pattern = re.compile(f"{col}=([^/]+)")

                    # Create a mapping of files to their extracted values
                    file_to_value = {}
                    for file_path in parquet_files:
                        match = pattern.search(file_path)
                        if match:
                            value = match.group(1)
                            # Convert to appropriate type
                            if col in ["year", "month"]:
                                value = int(value)
                            file_to_value[file_path] = value

                    if file_to_value:
                        # We'd need to map each row to its source file to assign the correct value
                        # This is a simplified approach that works if all rows in a file have the same value
                        # For a complete solution, we'd need to track source file for each row during read
                        self.logger.warning(
                            f"Adding '{col}' as a constant for each file - this is an approximation"
                        )

                        # For demonstration, just add the most common value if we can't match precisely
                        if len(set(file_to_value.values())) == 1:
                            # All values are the same, so we can just use that value
                            df[col] = list(file_to_value.values())[0]
                        else:
                            self.logger.warning(
                                f"Multiple values found for '{col}', cannot accurately restore"
                            )
                            # If you later need to implement proper path-based partition extraction,
                            # you would need to track which file each row came from during reading

            self.logger.info(
                f"Read {len(df)} rows from partitioned dataset in {base_dir}"
            )
            return df
        except Exception as e:
            self.logger.error(f"Error reading partitioned dataset: {e}")
            # Try the fallback approach using DuckDB directly
            try:
                import duckdb

                self.logger.info(
                    "Attempting fallback with DuckDB for reading partitioned dataset"
                )
                conn = duckdb.connect(":memory:")
                conn.execute("INSTALL parquet; LOAD parquet;")
                # Convert the DuckDB result to a polars DataFrame
                pandas_df = conn.execute(
                    f"SELECT * FROM parquet_scan('{base_dir}/**/*.parquet', UNION_BY_NAME=TRUE)"
                ).fetchdf()
                df = pl.from_pandas(pandas_df)
                conn.close()
                self.logger.info(
                    f"Successfully read {len(df)} rows using DuckDB fallback"
                )
                return df
            except Exception as e2:
                self.logger.error(f"DuckDB fallback also failed: {e2}")
                raise ValueError(
                    f"Could not read partitioned dataset: {e}. DuckDB fallback failed: {e2}"
                )


class DatabaseManager:
    """Facade around DuckDB for metadata and query operations.

    Manages a DuckDB connection, creates a canonical `market_data` view over
    Parquet files, and provides convenience helpers for typical queries used by
    the engine and CLI.
    """

    def __init__(
        self,
        db_path: str = "db/vegas.duckdb",
        parquet_dir: str = "db",
        test_mode: bool = False,
    ):
        """Initialize a DuckDB-backed database manager.

        :param db_path: Path to DuckDB database file (or ":memory:" in tests).
        :type db_path: str
        :param parquet_dir: Directory containing partitioned Parquet datasets.
        :type parquet_dir: str
        :param test_mode: If True, use in-memory DB and temp directory for Parquet.
        :type test_mode: bool
        """
        self.logger = logging.getLogger("vegas.database")
        # Use the logger configured by CLI - don't add handlers

        # Check for environment variable to enforce test mode
        if os.environ.get("VEGAS_TEST_MODE") == "1":
            test_mode = True
            self.logger.info(
                "Test mode enforced by VEGAS_TEST_MODE environment variable"
            )

        self.test_mode = test_mode

        # Use in-memory database and temporary directory in test mode
        if test_mode:
            self.db_path = ":memory:"
            self.parquet_dir = os.path.join(
                tempfile.mkdtemp(prefix="vegas_test_db_"), "parquet"
            )
            self.logger.info(
                f"Test mode enabled: Using in-memory database and temporary directory {self.parquet_dir}"
            )
        else:
            self.db_path = db_path
            self.parquet_dir = parquet_dir

        # Initialize managers and connection
        self.parquet_manager = ParquetManager(self.parquet_dir)
        self.conn = None
        self.connect()  # This will call _initialize_schema internally

        self.logger.info(f"DatabaseManager initialized with database: {self.db_path}")

    def connect(self) -> None:
        """Connect to DuckDB and initialize session pragmas and schema."""
        try:
            self.conn = duckdb.connect(self.db_path)
            self.logger.info(f"Connected to DuckDB database at {self.db_path}")

            # Enable automatic loading of Parquet files
            self.conn.execute("INSTALL parquet; LOAD parquet;")

            # Configure timestamp handling
            self.conn.execute("SET timezone='UTC';")
            self.conn.execute("SET default_null_order='nulls_first';")

            # Configure Parquet reading
            self.conn.execute("PRAGMA enable_object_cache;")
            self.conn.execute("SET enable_progress_bar=false;")

            # Configure threads
            self.conn.execute(f"PRAGMA threads={os.cpu_count()}")

            # Initialize database schema
            self._initialize_schema()

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def _initialize_schema(self) -> None:
        """Create base tables and materialize the `market_data` view if possible."""
        if not self.conn:
            self.connect()

        try:
            # Create symbols table
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS symbols (
                symbol VARCHAR PRIMARY KEY,
                first_date TIMESTAMP,
                last_date TIMESTAMP,
                data_points INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create data sources table
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS data_sources_id_seq;")
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY DEFAULT nextval('data_sources_id_seq'),
                source_name VARCHAR,
                source_path VARCHAR,
                format VARCHAR,
                row_count INTEGER,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create ingested_files table to track files and prevent duplicates
            self.conn.execute(
                """
            CREATE TABLE IF NOT EXISTS ingested_files (
                file_path VARCHAR PRIMARY KEY,
                file_hash VARCHAR,
                source_name VARCHAR,
                file_size INTEGER,
                row_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create market data view
            self._create_market_data_view()

            self.logger.info("Database schema initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}")
            # Create empty view as fallback
            self._create_empty_market_data_view()

    def _create_market_data_view(self) -> None:
        """Create or refresh a view over the partitioned Parquet files."""
        partitioned_dir = os.path.join(self.parquet_dir, "partitioned")

        # Check if partitioned directory exists and has Parquet files
        if os.path.exists(partitioned_dir) and glob.glob(
            os.path.join(partitioned_dir, "**/*.parquet"), recursive=True
        ):
            try:
                # First, explicitly set timestamp handling configuration
                self.conn.execute("SET timezone='UTC';")
                self.conn.execute("SET default_null_order='nulls_first';")

                # Create a temporary table with schema but no data
                self.conn.execute(
                    """
                CREATE OR REPLACE TABLE temp_market_data AS
                SELECT
                    CAST(NULL AS TIMESTAMP) as timestamp,
                    CAST(NULL AS VARCHAR) as symbol,
                    CAST(NULL AS DOUBLE) as open,
                    CAST(NULL AS DOUBLE) as high,
                    CAST(NULL AS DOUBLE) as low,
                    CAST(NULL AS DOUBLE) as close,
                    CAST(NULL AS DOUBLE) as volume
                WHERE 1=0
                """
                )

                # Create view based on the standardized table
                self.conn.execute(
                    """
                CREATE OR REPLACE VIEW market_data AS
                SELECT * FROM temp_market_data
                """
                )

                # Try to redefine the view to include real data
                try:
                    self.conn.execute(
                        f"""
                    CREATE OR REPLACE VIEW market_data AS
                    SELECT
                        timestamp::TIMESTAMP as timestamp,
                        symbol::VARCHAR as symbol,
                        open::DOUBLE as open,
                        high::DOUBLE as high,
                        low::DOUBLE as low,
                        close::DOUBLE as close,
                        volume::DOUBLE as volume
                    FROM parquet_scan('{partitioned_dir}/**/*.parquet', UNION_BY_NAME=TRUE)
                    """
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to create view over partitioned data: {e}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to create market_data view: {e}")
                # Create empty view as fallback
                self._create_empty_market_data_view()
        else:
            # No partitioned data yet, create empty view
            self._create_empty_market_data_view()

    def _create_empty_market_data_view(self) -> None:
        """Create an empty `market_data` view with a stable schema."""
        self.conn.execute(
            """
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
        """
        )
        self.logger.info("Created empty market_data view")

    def close(self) -> None:
        """Close the database connection if open."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")

    def __del__(self):
        """Clean up any temporary resources (test tempdirs) on deletion."""
        self.close()

        # Clean up temporary directories when in test mode
        if (
            hasattr(self, "test_mode")
            and self.test_mode
            and hasattr(self, "parquet_dir")
            and "vegas_test_db_" in self.parquet_dir
        ):
            try:
                # Get the parent directory (one level up from parquet_dir)
                temp_dir = os.path.dirname(self.parquet_dir)
                if os.path.exists(temp_dir) and "vegas_test_db_" in temp_dir:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned up temporary test directory: {temp_dir}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to clean up test directory {self.parquet_dir}: {e}"
                )

    def execute_query(
        self, query: str, parameters: tuple = ()
    ) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query and return a DuckDB relation.

        :param query: SQL query string.
        :type query: str
        :param parameters: Optional parameter tuple.
        :type parameters: tuple
        :returns: DuckDB relation containing the results.
        :rtype: duckdb.DuckDBPyRelation
        """
        if not self.conn:
            self.connect()

        try:
            return self.conn.execute(query, parameters)
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}\nQuery: {query}")
            raise

    def query_to_df(
        self, query: str, parameters: tuple = (), timezone: str = None
    ) -> pl.DataFrame:
        """Execute SQL and return results as a Polars DataFrame.

        :param query: SQL query string.
        :type query: str
        :param parameters: Optional parameter tuple.
        :type parameters: tuple
        :param timezone: Optional timezone name for timestamp columns.
        :type timezone: Optional[str]
        :returns: Polars DataFrame with query results.
        :rtype: pl.DataFrame
        """
        try:
            result = self.execute_query(query, parameters)
            # Convert the DuckDB result to a polars DataFrame
            df = result.pl()

            # Apply timezone to timestamp column if requested and it exists
            if timezone and "timestamp" in df.columns:
                # Check if timestamp column is a datetime
                if df.schema["timestamp"] == pl.Datetime:
                    df = df.with_columns(
                        pl.col("timestamp").cast(pl.Datetime("us", time_zone=timezone))
                    )
                    self.logger.debug(
                        f"Applied timezone {timezone} to timestamp column"
                    )
            return df
        except Exception as e:
            self.logger.error(f"Failed to convert query results to DataFrame: {e}")
            return pl.DataFrame()

    def ingest_data(self, df: pl.DataFrame, source_name: str) -> int:
        """Ingest market data into partitioned Parquet and register source.

        :param df: DataFrame with at least ['timestamp', 'symbol'] columns.
        :type df: pl.DataFrame
        :param source_name: Name/identifier for the source file or stream.
        :type source_name: str
        :returns: Number of rows ingested.
        :rtype: int
        :raises ValueError: If required columns are missing.
        """
        if df.is_empty():
            self.logger.warning(
                f"Empty DataFrame provided for source {source_name}, nothing to ingest"
            )
            return 0

        required_columns = ["timestamp", "symbol"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in data for source {source_name}"
                )

        try:
            # Ensure timestamp is a datetime with consistent timezone handling
            if df.schema["timestamp"].dtype != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

            # Normalize timestamps - convert to timezone-naive UTC for consistency
            timestamp_dtype = df.schema["timestamp"].dtype
            if timestamp_dtype.has_time_zone():
                # Convert to UTC and remove timezone info
                df = df.with_columns(
                    pl.col("timestamp").dt.convert_time_zone("UTC").dt.cast(pl.Datetime)
                )
            else:
                # Assume UTC if no timezone info is present
                self.logger.info("Timestamp data has no timezone info, assuming UTC")

            # Add year and month columns for partitioning
            df = df.with_columns(
                [
                    pl.col("timestamp").dt.year().alias("year"),
                    pl.col("timestamp").dt.month().alias("month"),
                ]
            )

            # Add partition path
            source_path = f"ingested/{source_name.replace('.', '_')}"

            # Write partitioned data - partition by year and month first to reduce partitions
            partition_cols = ["year", "month"]

            self.parquet_manager.write_data_partitioned(df, partition_cols)

            # Record the data source
            min_date = df.select(pl.col("timestamp").min()).item()
            max_date = df.select(pl.col("timestamp").max()).item()

            # Insert or update data source record
            self.conn.execute(
                """
            INSERT INTO data_sources (source_name, source_path, format, row_count, start_date, end_date)
            VALUES (?, ?, 'parquet', ?, ?, ?)
            """,
                (source_name, source_path, len(df), min_date, max_date),
            )

            # Insert symbols if they don't exist
            symbols = df.select("symbol").unique().to_series().to_list()
            for symbol in symbols:
                self.conn.execute(
                    """
                INSERT INTO symbols (symbol) VALUES (?)
                ON CONFLICT (symbol) DO NOTHING
                """,
                    (symbol,),
                )

            # Refresh the market_data view to include the new data
            self._create_market_data_view()

            self.logger.info(f"Ingested {len(df)} rows from {source_name}")
            return len(df)

        except Exception as e:
            self.logger.error(f"Data ingestion failed for {source_name}: {e}")
            raise

    def ingest_ohlcv_file(self, file_path: str) -> int:
        """Ingest a `.ohlcv-1h.csv.zst` file into the database.

        :param file_path: Absolute or relative path to an OHLCV file.
        :type file_path: str
        :returns: Number of rows ingested (0 if duplicate or invalid).
        :rtype: int
        :raises FileNotFoundError: If the file cannot be found.
        :raises ValueError: If the file format or columns are invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"OHLCV file not found: {file_path}")

        if not file_path.endswith(".ohlcv-1h.csv.zst"):
            raise ValueError(
                f"Invalid OHLCV file format for {file_path}. Expected .ohlcv-1h.csv.zst"
            )

        # Normalize the file path to ensure consistent comparison
        abs_file_path = os.path.abspath(file_path)
        file_size = os.path.getsize(abs_file_path)
        source_name = os.path.basename(abs_file_path)

        # Check if this file has already been ingested
        already_ingested = self.query_to_df(
            """
        SELECT file_path FROM ingested_files WHERE file_path = ?
        """,
            (abs_file_path,),
        )

        if not already_ingested.empty:
            self.logger.info(f"Skipping already ingested file: {abs_file_path}")
            return 0

        self.logger.info(f"Ingesting OHLCV file: {abs_file_path}")

        try:
            # Compute a simple hash for the file
            import hashlib

            file_hash = None
            with open(abs_file_path, "rb") as f:
                # Just hash the first 1MB to save time while still detecting changes
                file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()

            # Check if a file with the same hash has been ingested
            hash_match = self.query_to_df(
                """
            SELECT file_path FROM ingested_files WHERE file_hash = ?
            """,
                (file_hash,),
            )

            if not hash_match.empty:
                self.logger.info(
                    f"Skipping file with duplicate content: {abs_file_path}, matches {hash_match.iloc[0]['file_path']}"
                )
                return 0

            # Decompress the file
            import io

            import zstandard as zstd

            with open(abs_file_path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                stream_reader = dctx.stream_reader(f)
                text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

                # Load CSV data
                df = pl.read_csv(text_stream)

                # Validate required columns
                required_columns = [
                    "ts_event",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    raise ValueError(
                        f"Missing required columns in OHLCV file: {missing_columns}"
                    )

                # Rename ts_event to timestamp for consistency
                df = df.rename({"ts_event": "timestamp"})

                # Ensure timestamp is datetime
                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime).dt.replace_time_zone("UTC")
                )

                # Ingest the data
                row_count = self.ingest_data(df, source_name)

                # Record the file in ingested_files table
                self.conn.execute(
                    """
                INSERT INTO ingested_files (file_path, file_hash, source_name, file_size, row_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                    (abs_file_path, file_hash, source_name, file_size, row_count),
                )

                return row_count

        except Exception as e:
            self.logger.error(f"Failed to ingest OHLCV file {abs_file_path}: {e}")
            raise

    def ingest_ohlcv_directory(
        self,
        directory: str = "data",
        pattern: str = "*.ohlcv-1h.csv.zst",
        max_files: int = None,
    ) -> int:
        """Ingest all matching OHLCV files in a directory.

        :param directory: Directory to search recursively.
        :type directory: str
        :param pattern: Glob pattern (e.g., '*.ohlcv-1h.csv.zst').
        :type pattern: str
        :param max_files: Optional cap on number of files to ingest.
        :type max_files: Optional[int]
        :returns: Total ingested rows across new files.
        :rtype: int
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
            self.logger.warning(
                f"No OHLCV files found in {directory} with pattern {pattern}"
            )
            return 0

        # Limit the number of files if specified
        if max_files:
            files = files[:max_files]

        self.logger.info(f"Found {len(files)} OHLCV files to process")

        # Check which files are already ingested
        abs_file_paths = [os.path.abspath(f) for f in files]
        file_paths_str = "', '".join(abs_file_paths)

        already_ingested = set()
        if file_paths_str:
            query = f"SELECT file_path FROM ingested_files WHERE file_path IN ('{file_paths_str}')"
            try:
                result = self.query_to_df(query)
                already_ingested = set(
                    result["file_path"].tolist() if not result.empty else []
                )
            except Exception as e:
                # Fall back to individual checks if the combined query is too large
                self.logger.warning(f"Falling back to individual file checks: {e}")
                for file_path in abs_file_paths:
                    result = self.query_to_df(
                        "SELECT file_path FROM ingested_files WHERE file_path = ?",
                        (file_path,),
                    )
                    if not result.empty:
                        already_ingested.add(file_path)

        skipped_count = 0
        new_files = []
        for file_path in files:
            abs_path = os.path.abspath(file_path)
            if abs_path in already_ingested:
                skipped_count += 1
            else:
                new_files.append(file_path)

        if skipped_count > 0:
            self.logger.info(f"Skipping {skipped_count} already ingested files")

        if not new_files:
            self.logger.info("All files have already been ingested, nothing to do")
            return 0

        self.logger.info(f"Ingesting {len(new_files)} new OHLCV files")

        # Ingest each file
        total_rows = 0
        success_count = 0
        error_count = 0

        for file_path in new_files:
            try:
                rows = self.ingest_ohlcv_file(file_path)
                if rows > 0:
                    total_rows += rows
                    success_count += 1
                    self.logger.info(f"Successfully ingested {file_path}: {rows} rows")
                else:
                    skipped_count += 1
                    self.logger.info(
                        f"Skipped {file_path}: already ingested or duplicate content"
                    )
            except Exception as e:
                error_count += 1
                self.logger.error(f"Failed to ingest {file_path}: {e}")

        self.logger.info(
            f"Ingestion summary: {success_count} new files processed successfully, "
            f"{skipped_count} files skipped (duplicates), {error_count} errors, "
            f"{total_rows} total rows ingested"
        )
        return total_rows

    def get_available_trading_days(self) -> pl.Series:
        """Return all UTC trading days with at least one `market_data` row.

        :returns: Polars Date Series sorted ascending (empty if none).
        :rtype: pl.Series
        """
        try:
            df = self.query_to_df(
                """
                SELECT DISTINCT CAST(DATE_TRUNC('day', timestamp) AS DATE) AS date
                FROM market_data
                ORDER BY date
                """
            )
            if df.is_empty() or "date" not in df.columns:
                return pl.Series("date", [], dtype=pl.Date)
            # Ensure dtype is pl.Date
            if df.schema["date"] != pl.Date:
                df = df.with_columns(pl.col("date").cast(pl.Date))
            return df.get_column("date")
        except Exception as e:
            self.logger.error(f"Failed to get available trading days: {e}")
            return pl.Series("date", [], dtype=pl.Date)

    def get_available_dates(self) -> pl.DataFrame:
        """Return earliest and latest timestamps and unique day count.

        :returns: DataFrame with start_date, end_date, day_count.
        :rtype: pl.DataFrame
        """
        try:
            result = self.query_to_df(
                """
            SELECT
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as day_count
            FROM market_data
            """
            )

            # Handle case where no data is available
            if result.is_empty():
                return pl.DataFrame(
                    {"start_date": [None], "end_date": [None], "day_count": [0]}
                )

            return result
        except Exception as e:
            self.logger.error(f"Failed to get available dates: {e}")
            return pl.DataFrame(
                {"start_date": [None], "end_date": [None], "day_count": [0]}
            )

    def get_available_symbols(self) -> pl.DataFrame:
        """Return all symbols with their record counts.

        :returns: DataFrame with columns ['symbol', 'record_count'].
        :rtype: pl.DataFrame
        """
        try:
            return self.query_to_df(
                """
            SELECT
                symbol,
                COUNT(*) as record_count
            FROM market_data
            GROUP BY symbol
            ORDER BY symbol
            """
            )
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return pl.DataFrame(schema={"symbol": pl.Utf8, "record_count": pl.Int64})

    def get_market_data(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        symbols: List[str] = None,
        timezone: str = "UTC",
        bar_count: int = None,
    ) -> pl.DataFrame:
        """Query market data with flexible time and symbol filters.

        :param start_date: Optional inclusive start timestamp.
        :type start_date: Optional[datetime]
        :param end_date: Optional inclusive end timestamp.
        :type end_date: Optional[datetime]
        :param symbols: Optional list of symbols to restrict results.
        :type symbols: Optional[list[str]]
        :param timezone: Timezone for timestamp columns in the result.
        :type timezone: str
        :param bar_count: If provided, return the last `bar_count` rows per symbol up to `end_date`.
        :type bar_count: Optional[int]
        :returns: Market data DataFrame ordered by timestamp and symbol.
        :rtype: pl.DataFrame
        :raises ValueError: If `bar_count` is set without `end_date`.
        """
        try:
            if bar_count is not None:
                if end_date is None:
                    raise ValueError("end_date must be specified when using bar_count")

                conditions = ["timestamp <= ?"]
                params = [end_date]

                if symbols:
                    if len(symbols) <= 20:
                        assets_str = ", ".join([f"'{s}'" for s in symbols])
                        conditions.append(f"symbol IN ({assets_str})")
                    else:
                        self.conn.execute(
                            "CREATE TEMP TABLE IF NOT EXISTS temp_hist_symbols (symbol VARCHAR)"
                        )
                        self.conn.execute("DELETE FROM temp_hist_symbols")
                        self.conn.executemany(
                            "INSERT INTO temp_hist_symbols VALUES (?)",
                            [(s,) for s in symbols],
                        )
                        conditions.append(
                            "symbol IN (SELECT symbol FROM temp_hist_symbols)"
                        )

                where_str = " AND ".join(conditions)

                query = f"""
                WITH ranked_data AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM market_data
                    WHERE {where_str}
                )
                SELECT *
                FROM ranked_data
                WHERE rn <= ?
                ORDER BY timestamp ASC, symbol
                """

                final_params = tuple(params + [bar_count])
                return self.query_to_df(query, final_params, timezone=timezone)

            else:
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
                    if len(symbols) <= 10:
                        symbols_list = ", ".join([f"'{s}'" for s in symbols])
                        conditions.append(f"symbol IN ({symbols_list})")
                    else:
                        self.conn.execute(
                            "CREATE TEMP TABLE IF NOT EXISTS temp_symbols (symbol VARCHAR)"
                        )
                        self.conn.execute("DELETE FROM temp_symbols")
                        for symbol in symbols:
                            self.conn.execute(
                                "INSERT INTO temp_symbols VALUES (?)", (symbol,)
                            )
                        conditions.append("symbol IN (SELECT symbol FROM temp_symbols)")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp, symbol"

                return self.query_to_df(query, tuple(params), timezone=timezone)

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return pl.DataFrame()

    def get_unified_timestamps(
        self, start_date: datetime, end_date: datetime, timezone: str = "UTC"
    ) -> pl.Series:
        """Return a unique, sorted Series of timestamps across `market_data`.

        :param start_date: Inclusive lower bound.
        :type start_date: datetime
        :param end_date: Inclusive upper bound.
        :type end_date: datetime
        :param timezone: Target timezone for returned timestamps.
        :type timezone: str
        :returns: Series of timezone-aware datetimes sorted ascending.
        :rtype: pl.Series
        """
        try:
            df = self.query_to_df(
                """
                SELECT DISTINCT timestamp
                FROM market_data
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                """,
                (start_date, end_date),
                timezone=timezone,
            )
            if df.is_empty() or "timestamp" not in df.columns:
                return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=timezone))

            # Ensure we have timezone on the column
            ts_col = df.get_column("timestamp")
            if ts_col.dtype != pl.Datetime or (
                hasattr(ts_col.dtype, "time_zone")
                and ts_col.dtype.time_zone != timezone
            ):
                df = df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us", time_zone=timezone))
                )

            # Return as a Series unique & sorted (already sorted by query)
            return df.get_column("timestamp")
        except Exception as e:
            self.logger.error(f"Failed to get unified timestamps: {e}")
            return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=timezone))

    def get_data_sources(self) -> pl.DataFrame:
        """Return metadata for known data sources.

        :returns: DataFrame with data source records.
        :rtype: pl.DataFrame
        """
        return self.query_to_df("SELECT * FROM data_sources")

    def get_database_size(self) -> int:
        """Return the DuckDB file size in bytes (0 if absent)."""
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0

    def cleanup_duplicate_data(self) -> int:
        """Detect duplicates and provide guidance for de-duplication workflow.

        :returns: Estimated number of duplicate rows detected.
        :rtype: int
        """
        try:
            self.logger.info("Starting duplicate data cleanup...")

            # Create a temporary table with deduplicated data
            self.conn.execute(
                """
            CREATE TEMP TABLE deduplicated_market_data AS
            SELECT DISTINCT * FROM market_data
            """
            )

            # Get the count of rows before deduplication
            before_count = self.query_to_df(
                "SELECT COUNT(*) as count FROM market_data"
            ).iloc[0]["count"]

            # Get the count after deduplication
            after_count = self.query_to_df(
                "SELECT COUNT(*) as count FROM deduplicated_market_data"
            ).iloc[0]["count"]

            # Calculate duplicates removed
            duplicates_removed = before_count - after_count

            if duplicates_removed > 0:
                self.logger.info(f"Found {duplicates_removed} duplicate rows to remove")

                # This requires modifying the underlying Parquet files, which is complex
                # For simplicity, we'll log that duplicates were found and suggest a re-import
                self.logger.warning(
                    "To remove duplicates, a full data re-import may be required"
                )
                self.logger.info("Consider using the following steps:")
                self.logger.info("1. Export unique data to a backup")
                self.logger.info("2. Clear the database")
                self.logger.info("3. Re-import the unique data")

                # Export deduplicated data if needed
                # self._export_deduplicated_data()
            else:
                self.logger.info("No duplicate data found in the database")

            # Clean up
            self.conn.execute("DROP TABLE deduplicated_market_data")

            return duplicates_removed
        except Exception as e:
            self.logger.error(f"Failed to clean up duplicate data: {e}")
            return 0

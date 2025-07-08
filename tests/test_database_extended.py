"""Extended tests for the database functionality of the Vegas backtesting engine."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import glob
import io
import zstandard as zstd
from datetime import datetime, timedelta
from pathlib import Path

from vegas.database import DatabaseManager, ParquetManager


def generate_test_data(symbols=None, days=5, frequency='1h', with_gaps=False):
    """Generate test market data.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days to generate data for
        frequency: Data frequency ('1d', '1h', '1m')
        with_gaps: Whether to include data gaps
        
    Returns:
        DataFrame with test data
    """
    if symbols is None:
        symbols = ["TEST1", "TEST2", "TEST3"]
    
    # Generate timestamps
    base_date = datetime(2022, 1, 1)
    timestamps = []
    
    if frequency == '1d':
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            timestamps.append(current_date)
    elif frequency == '1h':
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            for hour in range(9, 16):  # Trading hours
                timestamps.append(current_date.replace(hour=hour))
    elif frequency == '1m':
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            for hour in range(9, 16):  # Trading hours
                for minute in range(0, 60, 5):  # Every 5 minutes
                    timestamps.append(current_date.replace(hour=hour, minute=minute))
    
    # Generate data
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for ts in timestamps:
            # Skip some data points if with_gaps is True
            if with_gaps and np.random.random() < 0.1:
                continue
                
            # Generate random price movement
            price_change = np.random.normal(0, 0.5)
            current_price = max(1, base_price * (1 + price_change / 100))
            
            # Add some volatility
            high = current_price * (1 + np.random.uniform(0, 0.5) / 100)
            low = current_price * (1 - np.random.uniform(0, 0.5) / 100)
            
            # Ensure high >= close >= low
            high = max(high, current_price)
            low = min(low, current_price)
            
            # Generate volume
            volume = np.random.randint(1000, 10000)
            
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": base_price,
                "high": high,
                "low": low,
                "close": current_price,
                "volume": volume
            })
            
            # Update base price for next interval
            base_price = current_price
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_csv_file(df, file_path):
    """Create a CSV file from a DataFrame."""
    df.to_csv(file_path, index=False)
    return file_path


def create_compressed_file(df, file_path):
    """Create a compressed CSV file from a DataFrame."""
    csv_data = df.to_csv(index=False)
    
    with open(file_path, 'wb') as f:
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(csv_data.encode('utf-8'))
        f.write(compressed_data)
    
    return file_path


def test_database_initialization(temp_db_dir):
    """Test database initialization and schema creation."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Verify tables are created
    tables = db.query_to_df("SHOW TABLES")
    assert not tables.empty
    
    # Check for required tables
    table_names = tables['name'].tolist()
    assert 'symbols' in table_names
    assert 'data_sources' in table_names
    
    # Check if market_data view exists - use a different approach
    try:
        # Just try to query the view (will raise exception if it doesn't exist)
        db.query_to_df("SELECT * FROM market_data LIMIT 1")
        view_exists = True
    except Exception:
        view_exists = False
    
    assert view_exists, "market_data view should exist"


def test_data_ingestion_formats(temp_db_dir):
    """Test data ingestion from different file formats."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Generate test data
    df = generate_test_data(symbols=["TEST1"], days=2)
    
    # Test CSV ingestion
    csv_path = os.path.join(temp_db_dir, "test_data.csv")
    create_csv_file(df, csv_path)
    rows_csv = db.ingest_data(df, "csv_source")
    assert rows_csv == len(df)
    
    # Test compressed file ingestion
    compressed_path = os.path.join(temp_db_dir, "test_data.csv.zst")
    create_compressed_file(df, compressed_path)
    
    # Read compressed file and ingest
    with open(compressed_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        data_buffer = dctx.decompress(f.read())
        csv_data = io.StringIO(data_buffer.decode('utf-8'))
        df_compressed = pd.read_csv(csv_data, parse_dates=['timestamp'])
    
    rows_compressed = db.ingest_data(df_compressed, "compressed_source")
    assert rows_compressed == len(df_compressed)
    
    # Test Parquet ingestion
    parquet_path = os.path.join(temp_db_dir, "test_data.parquet")
    df.to_parquet(parquet_path, index=False)
    df_parquet = pd.read_parquet(parquet_path)
    rows_parquet = db.ingest_data(df_parquet, "parquet_source")
    assert rows_parquet == len(df_parquet)
    
    # Verify all data is in the database
    total_rows = rows_csv + rows_compressed + rows_parquet
    result = db.query_to_df("SELECT COUNT(*) as count FROM market_data")
    assert result["count"].iloc[0] == total_rows


def test_partitioned_storage(temp_db_dir):
    """Test partitioned data storage and retrieval."""
    # Initialize ParquetManager
    parquet_dir = os.path.join(temp_db_dir, "parquet")
    pm = ParquetManager(parquet_dir)
    
    # Generate test data with multiple years and symbols
    symbols = ["TEST1", "TEST2", "TEST3"]
    dfs = []
    
    for year in [2020, 2021, 2022]:
        base_date = datetime(year, 1, 1)
        df = generate_test_data(symbols=symbols, days=5)
        # Adjust timestamps to the correct year
        df['timestamp'] = df['timestamp'].apply(lambda x: x.replace(year=year))
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add partition columns
    combined_df['year'] = combined_df['timestamp'].dt.year
    combined_df['month'] = combined_df['timestamp'].dt.month
    
    # Write partitioned data
    written_files = pm.write_data_partitioned(combined_df, partition_cols=['year', 'month', 'symbol'])
    
    # Verify files were created with correct structure
    assert len(written_files) > 0
    
    # Check partition structure
    for year in [2020, 2021, 2022]:
        year_dir = os.path.join(parquet_dir, "partitioned", f"year={year}")
        assert os.path.exists(year_dir)
        
        # Check for month directories
        month_dirs = glob.glob(os.path.join(year_dir, "month=*"))
        assert len(month_dirs) > 0
        
        # Check for symbol directories in first month
        first_month_dir = month_dirs[0]
        symbol_dirs = glob.glob(os.path.join(first_month_dir, "symbol=*"))
        assert len(symbol_dirs) > 0
    
    # Read partitioned dataset
    partitioned_df = pm.read_partitioned_dataset(os.path.join(parquet_dir, "partitioned"))
    
    # Verify data integrity
    assert len(partitioned_df) == len(combined_df)
    
    # Test reading with filters
    filters = [('year', '=', 2021)]
    filtered_df = pm.read_partitioned_dataset(
        os.path.join(parquet_dir, "partitioned"),
        filters=filters
    )
    assert len(filtered_df) < len(combined_df)
    assert all(filtered_df['year'] == 2021)


def test_market_data_queries(temp_db_dir):
    """Test market data queries with various filters."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Generate test data spanning multiple days
    symbols = ["TEST1", "TEST2", "TEST3"]
    df = generate_test_data(symbols=symbols, days=10)
    
    # Ingest data
    db.ingest_data(df, "test_source")
    
    # Test date range filtering
    start_date = df["timestamp"].min() + timedelta(days=2)
    end_date = df["timestamp"].max() - timedelta(days=2)
    
    date_filtered = db.get_market_data(start_date=start_date, end_date=end_date)
    assert not date_filtered.empty
    assert date_filtered["timestamp"].min() >= start_date
    assert date_filtered["timestamp"].max() <= end_date
    
    # Test symbol filtering
    symbol_filtered = db.get_market_data(symbols=["TEST1"])
    assert not symbol_filtered.empty
    assert set(symbol_filtered["symbol"].unique()) == {"TEST1"}
    
    # Test combined filtering
    combined_filtered = db.get_market_data(
        start_date=start_date,
        end_date=end_date,
        symbols=["TEST1", "TEST2"]
    )
    assert not combined_filtered.empty
    assert combined_filtered["timestamp"].min() >= start_date
    assert combined_filtered["timestamp"].max() <= end_date
    assert set(combined_filtered["symbol"].unique()) <= {"TEST1", "TEST2"}
    
    # Test aggregation query
    agg_query = """
    SELECT 
        symbol, 
        DATE_TRUNC('day', timestamp) as date,
        MIN(low) as day_low,
        MAX(high) as day_high,
        FIRST(open) as day_open,
        LAST(close) as day_close,
        SUM(volume) as day_volume
    FROM market_data
    GROUP BY symbol, DATE_TRUNC('day', timestamp)
    ORDER BY date, symbol
    """
    daily_data = db.query_to_df(agg_query)
    
    # Verify aggregation
    assert not daily_data.empty
    assert len(daily_data) < len(df)  # Should be fewer rows after aggregation
    assert set(daily_data["symbol"].unique()) == set(symbols)


def test_database_connection_handling(temp_db_dir):
    """Test database connection handling."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Test connection is active
    result = db.query_to_df("SELECT 1 as test")
    assert result["test"].iloc[0] == 1
    
    # Test closing the connection
    db.close()
    
    # Test reconnecting
    # Create a new instance instead of reusing the old one
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Verify connection works
    result = db.query_to_df("SELECT 2 as test")
    assert result["test"].iloc[0] == 2


def test_database_size_management(temp_db_dir):
    """Test database size management."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    db = DatabaseManager(db_path, temp_db_dir)
    
    # Get initial size
    initial_size = db.get_database_size()
    
    # Generate and ingest test data
    df1 = generate_test_data(symbols=["TEST1"], days=5)
    db.ingest_data(df1, "test_source_1")
    
    # Check size after first ingestion
    size_after_first = db.get_database_size()
    
    # Generate and ingest more test data
    df2 = generate_test_data(symbols=["TEST2", "TEST3"], days=10)
    db.ingest_data(df2, "test_source_2")
    
    # Check size after second ingestion
    size_after_second = db.get_database_size()
    
    # Verify size changes
    assert size_after_second >= size_after_first, "Database size should increase or stay the same after adding more data"


def test_parquet_format_conversions(temp_db_dir):
    """Test data integrity during format conversions."""
    # Initialize ParquetManager
    parquet_dir = os.path.join(temp_db_dir, "parquet")
    pm = ParquetManager(parquet_dir)
    
    # Generate test data with various data types
    df = generate_test_data(symbols=["TEST1"], days=2)
    
    # Add columns with different data types
    df["int_col"] = np.random.randint(0, 100, size=len(df))
    df["float_col"] = np.random.random(size=len(df))
    df["bool_col"] = np.random.choice([True, False], size=len(df))
    df["str_col"] = ["str_" + str(i) for i in range(len(df))]
    
    # Write to Parquet
    file_path = os.path.join(parquet_dir, "test_types.parquet")
    pm.write_dataframe_to_parquet(df, file_path)
    
    # Read back
    read_df = pm.read_parquet_file(file_path)
    
    # Verify data types are preserved
    assert read_df["int_col"].dtype == df["int_col"].dtype
    assert read_df["float_col"].dtype == df["float_col"].dtype
    assert read_df["bool_col"].dtype == df["bool_col"].dtype
    assert read_df["str_col"].dtype == df["str_col"].dtype
    
    # Verify timestamp column is correctly parsed
    assert pd.api.types.is_datetime64_any_dtype(read_df["timestamp"])
    
    # Verify data values are preserved
    pd.testing.assert_frame_equal(
        df.sort_values("timestamp").reset_index(drop=True),
        read_df.sort_values("timestamp").reset_index(drop=True)
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
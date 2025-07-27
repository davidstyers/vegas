"""Tests for database duplicate detection and prevention functionality."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import hashlib
from datetime import datetime, timedelta
import zstandard as zstd
import io
import duckdb

from vegas.database import DatabaseManager, ParquetManager


def generate_test_data(symbols=None, days=2):
    """Generate test market data.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days to generate data for
        
    Returns:
        DataFrame with test data
    """
    if symbols is None:
        symbols = ["TEST1", "TEST2"]
    
    # Generate timestamps
    base_date = datetime(2022, 1, 1)
    timestamps = []
    for i in range(days):
        current_date = base_date + timedelta(days=i)
        for hour in range(9, 16):  # Trading hours
            timestamps.append(current_date.replace(hour=hour))
    
    # Generate data
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for ts in timestamps:
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


def create_ohlcv_file(df, file_path):
    """Create a compressed OHLCV file from a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        file_path: Path to write the file
        
    Returns:
        Path to the created file
    """
    # Ensure we have the required columns
    df = df.rename(columns={"timestamp": "ts_event"})
    
    # Convert to CSV
    csv_data = df.to_csv(index=False)
    
    # Compress with zstd
    with open(file_path, 'wb') as f:
        cctx = zstd.ZstdCompressor()
        compressed_data = cctx.compress(csv_data.encode('utf-8'))
        f.write(compressed_data)
    
    return file_path


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_test_db_manager(temp_dir):
    """Create a DatabaseManager for testing with proper isolation.
    
    This ensures the test database is completely isolated from production.
    
    Args:
        temp_dir: Temporary directory for test files
        
    Returns:
        Initialized DatabaseManager for tests
    """
    # Create test DB path in the temporary directory
    db_path = os.path.join(temp_dir, "test.duckdb")
    
    # Make the dir_name variable accessible to DatabaseManager
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create the database manager in test mode for proper isolation
    db_manager = DatabaseManager(db_path, temp_dir, test_mode=True)
    
    return db_manager


def test_duplicate_file_detection(temp_db_dir):
    """Test detection and skipping of duplicate files."""
    # Initialize DatabaseManager with test isolation
    db = create_test_db_manager(temp_db_dir)
    
    # Generate test data
    df = generate_test_data()
    
    # Create first OHLCV file
    file_path1 = os.path.join(temp_db_dir, "test_data_1.ohlcv-1h.csv.zst")
    create_ohlcv_file(df, file_path1)
    
    # Ingest the file
    rows1 = db.ingest_ohlcv_file(file_path1)
    assert rows1 > 0, "First file should be ingested successfully"
    
    # Ingest the same file again - should be skipped
    rows2 = db.ingest_ohlcv_file(file_path1)
    assert rows2 == 0, "Duplicate file should be skipped"
    
    # Create another file with the same content but different name
    file_path2 = os.path.join(temp_db_dir, "test_data_2.ohlcv-1h.csv.zst")
    create_ohlcv_file(df, file_path2)
    
    # Ingest the file with same content - should be skipped based on hash
    rows3 = db.ingest_ohlcv_file(file_path2)
    assert rows3 == 0, "File with duplicate content should be skipped"
    
    # Check ingested_files table
    ingested_files = db.query_to_df("SELECT * FROM ingested_files")
    assert len(ingested_files) == 1, "Only one unique file should be recorded"
    
    # Clean up
    db.close()


def test_duplicate_directory_ingestion(temp_db_dir):
    """Test ingestion of a directory with duplicate files."""
    # Initialize DatabaseManager with test isolation
    db = create_test_db_manager(temp_db_dir)
    
    # Create a subdirectory for OHLCV files
    ohlcv_dir = os.path.join(temp_db_dir, "ohlcv_data")
    os.makedirs(ohlcv_dir, exist_ok=True)
    
    # Generate test data - three different datasets
    df1 = generate_test_data(symbols=["AAPL", "MSFT"])
    df2 = generate_test_data(symbols=["GOOG", "AMZN"])
    df3 = generate_test_data(symbols=["NFLX", "TSLA"])
    
    # Create OHLCV files
    file_path1 = os.path.join(ohlcv_dir, "data1.ohlcv-1h.csv.zst")
    file_path2 = os.path.join(ohlcv_dir, "data2.ohlcv-1h.csv.zst")
    file_path3 = os.path.join(ohlcv_dir, "data3.ohlcv-1h.csv.zst")
    
    create_ohlcv_file(df1, file_path1)
    create_ohlcv_file(df2, file_path2)
    create_ohlcv_file(df3, file_path3)
    
    # Duplicate one of the files
    file_path1_dup = os.path.join(ohlcv_dir, "data1_duplicate.ohlcv-1h.csv.zst")
    shutil.copy(file_path1, file_path1_dup)
    
    # Ingest the directory
    total_rows = db.ingest_ohlcv_directory(ohlcv_dir)
    
    # Check the data sources
    data_sources = db.query_to_df("SELECT * FROM data_sources")
    assert len(data_sources) == 3, "Only 3 unique sources should be ingested, not the duplicate"
    
    # Check ingested_files table
    ingested_files = db.query_to_df("SELECT * FROM ingested_files")
    assert len(ingested_files) == 3, "Only 3 unique files should be recorded"
    
    # Try ingesting the directory again - should skip all files
    rows_second_run = db.ingest_ohlcv_directory(ohlcv_dir)
    assert rows_second_run == 0, "Second run should skip all files"
    
    # Clean up
    db.close()


def test_content_based_duplicate_detection(temp_db_dir):
    """Test detection of files with identical content using MD5 hash."""
    # Initialize DatabaseManager with test isolation
    db = create_test_db_manager(temp_db_dir)
    
    # Generate test data
    df = generate_test_data()
    
    # Create first OHLCV file
    file_path1 = os.path.join(temp_db_dir, "test_data_original.ohlcv-1h.csv.zst")
    create_ohlcv_file(df, file_path1)
    
    # Ingest the first file
    db.ingest_ohlcv_file(file_path1)
    
    # Create a second file with same content but slightly modified path
    # This tests the hash-based detection
    file_path2 = os.path.join(temp_db_dir, "subfolder", "test_data_modified_path.ohlcv-1h.csv.zst")
    os.makedirs(os.path.dirname(file_path2), exist_ok=True)
    create_ohlcv_file(df, file_path2)
    
    # Ingest the second file - should be skipped based on hash
    rows = db.ingest_ohlcv_file(file_path2)
    assert rows == 0, "File with same content should be skipped based on MD5 hash"
    
    # Check the hash in the database
    hash_info = db.query_to_df("SELECT file_path, file_hash FROM ingested_files")
    assert len(hash_info) == 1, "Only one file hash should be recorded"
    
    # Generate slightly different data
    df_modified = df.copy()
    df_modified.iloc[0, df_modified.columns.get_loc("open")] += 0.01  # Small change
    
    # Create third file with slightly modified content
    file_path3 = os.path.join(temp_db_dir, "test_data_modified_content.ohlcv-1h.csv.zst")
    create_ohlcv_file(df_modified, file_path3)
    
    # Ingest the third file - should be ingested as content is different
    rows = db.ingest_ohlcv_file(file_path3)
    assert rows > 0, "File with different content should be ingested"
    
    # Verify two distinct file hashes in the database
    hash_info = db.query_to_df("SELECT file_path, file_hash FROM ingested_files")
    assert len(hash_info) == 2, "Two distinct file hashes should be recorded"
    assert len(hash_info["file_hash"].unique()) == 2, "Two unique hash values should exist"
    
    # Clean up
    db.close()


def test_duplicate_data_cleanup(temp_db_dir):
    """Test the duplicate data cleanup functionality."""
    # Initialize DatabaseManager with test isolation
    db = create_test_db_manager(temp_db_dir)
    
    # Generate test data
    df = generate_test_data()
    
    # Create a test table for detecting duplicates
    db.execute_query("""
    CREATE TABLE test_duplicates (
        timestamp TIMESTAMP,
        symbol VARCHAR,
        value FLOAT
    )
    """)
    
    # Insert some data with duplicates
    db.execute_query("""
    INSERT INTO test_duplicates VALUES
        ('2022-01-01 10:00:00', 'TEST1', 100.0),
        ('2022-01-01 10:00:00', 'TEST1', 100.0),  -- Duplicate
        ('2022-01-01 11:00:00', 'TEST1', 101.0),
        ('2022-01-01 11:00:00', 'TEST2', 200.0),
        ('2022-01-01 11:00:00', 'TEST2', 200.0)   -- Duplicate
    """)
    
    # Verify we have duplicates
    initial_count = db.query_to_df("SELECT COUNT(*) as count FROM test_duplicates").iloc[0]['count']
    unique_count = db.query_to_df("SELECT COUNT(*) as count FROM (SELECT DISTINCT * FROM test_duplicates)").iloc[0]['count']
    
    assert initial_count > unique_count, "Duplicates should exist in test table"
    assert initial_count == 5, "Should have 5 rows total"
    assert unique_count == 3, "Should have 3 unique rows"
    
    # Create deduplicated temp table similar to cleanup_duplicate_data method
    db.execute_query("""
    CREATE TEMP TABLE deduplicated_test AS
    SELECT DISTINCT * FROM test_duplicates
    """)
    
    # Count duplicates
    duplicate_count = initial_count - unique_count
    
    # Verify duplicate count
    assert duplicate_count == 2, "Should detect 2 duplicate rows"
    
    # Clean up
    db.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
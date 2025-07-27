"""Tests for the data layer functionality of the Vegas backtesting engine."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import glob
import io
import zstandard as zstd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vegas.data import DataLayer


def generate_test_data(symbols=None, days=5, frequency='1h', with_gaps=False, include_timezone=False):
    """Generate test market data.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days to generate data for
        frequency: Data frequency ('1d', '1h', '1m')
        with_gaps: Whether to include data gaps
        include_timezone: Whether to include timezone info in timestamps
        
    Returns:
        DataFrame with test data
    """
    if symbols is None:
        symbols = ["TEST1", "TEST2", "TEST3"]
    
    # Generate timestamps
    base_date = datetime(2022, 1, 1)
    if include_timezone:
        base_date = base_date.replace(tzinfo=timezone.utc)
        
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
def temp_data_dir():
    """Create a temporary directory for data files."""
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


def create_corrupt_csv(file_path):
    """Create a corrupt CSV file."""
    with open(file_path, 'w') as f:
        f.write("timestamp,symbol,open,high,low,close,volume\n")
        f.write("2022-01-01 09:00:00,TEST1,100,101,99,100.5,5000\n")
        f.write("not-a-date,TEST1,invalid,101,99,100.5,5000\n")
    return file_path


def test_data_loading_single_file(temp_data_dir, test_data_layer):
    """Test loading data from a single file."""
    # Use the test_data_layer fixture which is already in test mode
    data_layer = test_data_layer
    
    # Generate test data
    df = generate_test_data(symbols=["TEST1", "TEST2"], days=2)
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "test_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Verify data is loaded
    assert data_layer.data is not None
    assert not data_layer.data.empty
    assert len(data_layer.data) == len(df)
    assert set(data_layer.symbols) == set(df["symbol"].unique())
    
    # Test loading compressed file
    compressed_path = os.path.join(temp_data_dir, "test_data.csv.zst")
    create_compressed_file(df, compressed_path)
    
    # Create a new data layer in test mode
    data_layer2 = DataLayer(data_dir=temp_data_dir, test_mode=True)
    data_layer2.load_data(compressed_path)
    
    # Verify data is loaded
    assert data_layer2.data is not None
    assert not data_layer2.data.empty
    assert len(data_layer2.data) == len(df)
    
    # Clean up second data layer
    data_layer2.close()


def test_loading_multiple_files(temp_data_dir):
    """Test loading data from multiple files."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir)
    
    # Generate test data for multiple symbols
    symbols = ["TEST1", "TEST2", "TEST3"]
    dfs = []
    
    for symbol in symbols:
        df = generate_test_data(symbols=[symbol], days=2)
        file_path = os.path.join(temp_data_dir, f"{symbol}_data.csv")
        create_csv_file(df, file_path)
        dfs.append(df)
    
    # Create a directory for multiple files
    multi_dir = os.path.join(temp_data_dir, "multi")
    os.makedirs(multi_dir, exist_ok=True)
    
    # Create multiple files in the directory
    for i, symbol in enumerate(symbols):
        file_path = os.path.join(multi_dir, f"{symbol}_data.csv")
        create_csv_file(dfs[i], file_path)
    
    # Load data from directory
    data_layer.load_data(directory=multi_dir, file_pattern="*.csv")
    
    # Verify data is loaded
    assert data_layer.data is not None
    assert not data_layer.data.empty
    
    # Combined length should be sum of individual dataframes
    expected_length = sum(len(df) for df in dfs)
    assert len(data_layer.data) == expected_length
    
    # All symbols should be present
    assert set(data_layer.symbols) == set(symbols)


def test_data_validation_and_error_handling(temp_data_dir):
    """Test data validation and error handling."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir)
    
    # Create a file with missing required columns
    invalid_path = os.path.join(temp_data_dir, "invalid.csv")
    with open(invalid_path, 'w') as f:
        f.write("timestamp,price\n")  # Missing 'symbol' column
        f.write("2022-01-01 09:00:00,100\n")
    
    # Loading should raise ValueError for missing columns
    with pytest.raises(ValueError):
        data_layer.load_data(invalid_path)
    
    # Create a corrupt CSV file
    corrupt_path = os.path.join(temp_data_dir, "corrupt.csv")
    create_corrupt_csv(corrupt_path)
    
    # Loading should handle parsing errors
    with pytest.raises(Exception):
        data_layer.load_data(corrupt_path)
    
    # Test with non-existent file
    nonexistent_path = os.path.join(temp_data_dir, "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        data_layer.load_data(nonexistent_path)
    
    # Test with empty directory
    empty_dir = os.path.join(temp_data_dir, "empty")
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        data_layer.load_data(directory=empty_dir, file_pattern="*.csv")


def test_handling_missing_or_corrupt_data(temp_data_dir):
    """Test handling of missing or corrupt data."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir)
    
    # Generate test data with gaps
    df = generate_test_data(symbols=["TEST1"], days=5, with_gaps=True)
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "gapped_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Verify data is loaded despite gaps
    assert data_layer.data is not None
    assert not data_layer.data.empty
    
    # Test loading multiple files with one being corrupt
    valid_df = generate_test_data(symbols=["TEST2"], days=2)
    valid_path = os.path.join(temp_data_dir, "valid.csv")
    create_csv_file(valid_df, valid_path)
    
    corrupt_path = os.path.join(temp_data_dir, "corrupt.csv")
    create_corrupt_csv(corrupt_path)
    
    # This should load the valid file and skip/report the corrupt one
    with pytest.raises(Exception):
        data_layer.load_data(directory=temp_data_dir, file_pattern="*.csv")


def test_data_retrieval_for_specific_time_ranges(temp_data_dir):
    """Test data retrieval for specific time ranges."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir, test_mode=True)
    
    # Generate test data spanning multiple days
    df = generate_test_data(symbols=["TEST1", "TEST2"], days=10)
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "test_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Get data for a specific time range
    start_date = df["timestamp"].min() + timedelta(days=2)
    end_date = df["timestamp"].max() - timedelta(days=2)
    
    filtered_data = data_layer.get_data_for_backtest(start_date, end_date)
    
    # Verify filtered data
    assert not filtered_data.empty
    assert filtered_data["timestamp"].min() >= start_date
    assert filtered_data["timestamp"].max() <= end_date
    
    # Test with start date only
    start_only = data_layer.get_data_for_backtest(start_date, None)
    assert not start_only.empty
    assert start_only["timestamp"].min() >= start_date
    
    # Test with end date only
    end_only = data_layer.get_data_for_backtest(None, end_date)
    assert not end_only.empty
    assert end_only["timestamp"].max() <= end_date


def test_symbol_filtering_and_universe_selection(temp_data_dir):
    """Test symbol filtering and universe selection."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir)
    
    # Generate test data with multiple symbols
    symbols = ["TEST1", "TEST2", "TEST3", "TEST4", "TEST5"]
    df = generate_test_data(symbols=symbols, days=5)
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "test_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Get universe at a specific date
    mid_date = df["timestamp"].min() + (df["timestamp"].max() - df["timestamp"].min()) / 2
    universe = data_layer.get_universe(mid_date)
    
    # Verify universe
    assert set(universe) == set(symbols)
    
    # Get data for specific symbols
    filtered_symbols = ["TEST1", "TEST3"]
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    filtered_data = data_layer.get_data_for_backtest(start_date, end_date, filtered_symbols)
    
    # Verify filtered data
    assert not filtered_data.empty
    assert set(filtered_data["symbol"].unique()) == set(filtered_symbols)


def test_data_aggregation_and_resampling(temp_data_dir):
    """Test data aggregation and resampling."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir, test_mode=True)
    
    # Generate test data with minute frequency
    df = generate_test_data(symbols=["TEST1"], days=2, frequency='1m')
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "minute_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Get data info
    info = data_layer.get_data_info()
    
    # Verify minute data is loaded
    assert info["row_count"] > 0
    
    # If database is available, test aggregation query
    if data_layer.use_database and data_layer.db_manager:
        # Aggregate to hourly data
        hourly_query = """
        SELECT 
            symbol,
            DATE_TRUNC('hour', timestamp) as timestamp,
            FIRST(open) as open,
            MAX(high) as high,
            MIN(low) as low,
            LAST(close) as close,
            SUM(volume) as volume
        FROM market_data
        GROUP BY symbol, DATE_TRUNC('hour', timestamp)
        ORDER BY timestamp, symbol
        """
        
        hourly_data = data_layer.db_manager.query_to_df(hourly_query)
        
        # Verify aggregation
        assert not hourly_data.empty
        assert len(hourly_data) < len(df)  # Should be fewer rows after aggregation


def test_handling_timezone_information(temp_data_dir):
    """Test handling of timezone information."""
    # Initialize DataLayer
    data_layer = DataLayer(data_dir=temp_data_dir, test_mode=True)
    
    # Generate test data with timezone info
    df = generate_test_data(symbols=["TEST1"], days=2, include_timezone=True)
    
    # Create CSV file
    csv_path = os.path.join(temp_data_dir, "tz_data.csv")
    create_csv_file(df, csv_path)
    
    # Load data
    data_layer.load_data(csv_path)
    
    # Verify data is loaded
    assert data_layer.data is not None
    assert not data_layer.data.empty
    
    # Check timestamps - note that our database now normalizes timestamps to naive UTC
    first_ts = data_layer.data["timestamp"].iloc[0]
    
    # Test filtering with timezone-aware timestamps
    # Since the database converts to UTC naive, we need to do the same for comparison
    start_date_aware = df["timestamp"].min() + timedelta(hours=5)
    
    # Convert to naive UTC for comparison (same conversion the database does)
    if start_date_aware.tzinfo is not None:
        start_date = start_date_aware.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        start_date = start_date_aware
        
    filtered_data = data_layer.get_data_for_backtest(start_date_aware, None)
    
    # Verify filtered data
    assert not filtered_data.empty
    assert filtered_data["timestamp"].min() >= start_date  # Compare with naive UTC


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
"""Tests for the database functionality of the Vegas backtesting engine."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta

from vegas.database import DatabaseManager, ParquetManager


def generate_test_data(symbols=None, days=5):
    """Generate test market data.
    
    Args:
        symbols: List of symbols to generate data for
        days: Number of days to generate data for
        
    Returns:
        DataFrame with test data
    """
    if symbols is None:
        symbols = ["TEST1", "TEST2", "TEST3"]
    
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


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for database files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_parquet_manager(temp_db_dir):
    """Test the ParquetManager class."""
    # Initialize ParquetManager
    parquet_dir = os.path.join(temp_db_dir, "parquet")
    pm = ParquetManager(parquet_dir)
    
    # Generate test data
    df = generate_test_data(symbols=["TEST1"], days=2)
    
    # Test writing to Parquet file
    file_path = os.path.join(parquet_dir, "test.parquet")
    result_path = pm.write_dataframe_to_parquet(df, file_path)
    
    # Verify file was created
    assert os.path.exists(result_path)
    
    # Test reading from Parquet file
    read_df = pm.read_parquet_file(file_path)
    
    # Verify data integrity
    assert len(read_df) == len(df)
    assert set(read_df.columns) == set(df.columns)
    
    # Test partitioned writing
    df["year"] = df["timestamp"].dt.year
    written_files = pm.write_data_partitioned(df, partition_cols=["year", "symbol"])
    
    # Verify files were created
    assert len(written_files) > 0
    
    # Test reading partitioned dataset
    partition_dir = os.path.join(parquet_dir, "partitioned")
    partitioned_df = pm.read_partitioned_dataset(partition_dir)
    
    # Verify data integrity
    assert len(partitioned_df) == len(df)


def test_database_manager(temp_db_dir):
    """Test the DatabaseManager class."""
    # Initialize DatabaseManager
    db_path = os.path.join(temp_db_dir, "vegas.duckdb")
    parquet_dir = os.path.join(temp_db_dir, "parquet")
    db = DatabaseManager(db_path, parquet_dir)
    
    # Generate test data
    symbols = ["TEST1", "TEST2", "TEST3"]
    df = generate_test_data(symbols=symbols, days=3)
    
    # Test data ingestion
    rows_ingested = db.ingest_data(df, "test_source")
    assert rows_ingested == len(df)
    
    # Test querying data
    result = db.query_to_df("SELECT COUNT(*) as count FROM market_data")
    assert result["count"].iloc[0] == len(df)
    
    # Test symbol data
    symbols_df = db.get_available_symbols()
    assert len(symbols_df) == len(symbols)
    assert set(symbols_df["symbol"].tolist()) == set(symbols)
    
    # Test date range
    dates_df = db.get_available_dates()
    assert not dates_df.empty
    assert dates_df["start_date"].iloc[0] == df["timestamp"].min()
    assert dates_df["end_date"].iloc[0] == df["timestamp"].max()
    
    # Test filtered queries
    start_date = df["timestamp"].min() + timedelta(days=1)
    filtered_df = db.get_market_data(start_date=start_date)
    assert len(filtered_df) < len(df)
    assert filtered_df["timestamp"].min() >= start_date
    
    # Test symbol filtering
    symbol_df = db.get_market_data(symbols=["TEST1"])
    assert len(symbol_df) < len(df)
    assert set(symbol_df["symbol"].unique()) == {"TEST1"}
    
    # Test database size
    size = db.get_database_size()
    assert size > 0


def test_database_integration():
    """Test integration with the DataLayer class."""
    try:
        # Skip if dependencies not available
        import duckdb
        import pyarrow
    except ImportError:
        pytest.skip("DuckDB or PyArrow not available")
    
    # This test requires the data layer module
    from vegas.data import DataLayer
    
    # Use a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize DataLayer
        data_layer = DataLayer(data_dir=temp_dir)
        
        # Check if database initialized
        assert data_layer.use_database
        assert data_layer.db_manager is not None
        
        # Generate test data
        df = generate_test_data(symbols=["TEST1", "TEST2"], days=2)
        
        # Save to CSV
        csv_path = os.path.join(temp_dir, "test_data.csv")
        df.to_csv(csv_path, index=False)
        
        # Load data
        data_layer.load_data(csv_path)
        
        # Verify data is loaded in memory
        assert data_layer.data is not None
        assert not data_layer.data.empty
        
        # Verify data is ingested into database
        info = data_layer.get_data_info()
        assert info["row_count"] > 0
        assert info["symbol_count"] == 2
        
        # Test getting data for backtest
        start = df["timestamp"].min()
        end = df["timestamp"].max()
        backtest_data = data_layer.get_data_for_backtest(start, end)
        
        # Verify data integrity
        assert len(backtest_data) > 0
        assert backtest_data["timestamp"].min() >= start
        assert backtest_data["timestamp"].max() <= end


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
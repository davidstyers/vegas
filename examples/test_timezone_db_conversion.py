#!/usr/bin/env python3
"""
Test script to verify database-level timezone conversion.
"""

import polars as pl
from datetime import datetime, timedelta
from vegas.engine import BacktestEngine
from vegas.data import DataLayer
import time

# Configure different timezones to test
TIMEZONES = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]

def test_timezone_conversion():
    """Test the database-level timezone conversion."""
    
    print("Testing database-level timezone conversion...")
    
    # Create a sample data frame with market data
    print("Creating sample data...")
    data = {
        "timestamp": [
            datetime(2023, 1, 1, 9, 30),  # 9:30 AM
            datetime(2023, 1, 1, 16, 0),  # 4:00 PM
            datetime(2023, 1, 2, 9, 30),  # 9:30 AM next day
            datetime(2023, 1, 2, 16, 0)   # 4:00 PM next day
        ],
        "symbol": ["SPY", "SPY", "SPY", "SPY"],
        "open": [100.0, 102.0, 101.0, 103.0],
        "high": [102.0, 104.0, 103.0, 105.0],
        "low": [99.0, 101.0, 100.0, 102.0],
        "close": [101.0, 103.0, 102.0, 104.0],
        "volume": [1000, 1200, 1100, 1300]
    }
    
    df = pl.DataFrame(data)
    print(f"Sample data created: {len(df)} rows")
    
    # Test database-level timezone conversion with each timezone
    for tz in TIMEZONES:
        print(f"\n\n--- Testing with timezone: {tz} ---")
        
        # Create engine with the specific timezone
        engine = BacktestEngine(timezone=tz)
        data_layer = engine.data_layer
        
        # Use a temporary database
        data_layer.db_manager._create_market_data_view()
        
        # Ingest the sample data
        print(f"Ingesting data into database with timezone {tz}...")
        data_layer.ingest_to_database(df, f"test_timezone_{tz}")
        
        # Query the data back using the database's timezone conversion
        print(f"Querying data back with timezone {tz}...")
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 3)
        
        # 1. Query with timezone conversion in database
        start_time = time.time()
        data_db = data_layer.db_manager.get_market_data(
            start_date=start_date,
            end_date=end_date,
            timezone=tz
        )
        db_query_time = time.time() - start_time
        
        # 2. Query without timezone conversion in database
        start_time = time.time()
        data_no_tz = data_layer.db_manager.get_market_data(
            start_date=start_date,
            end_date=end_date
        )
        no_tz_query_time = time.time() - start_time
        
        # 3. Apply timezone conversion after query
        start_time = time.time()
        data_post_convert = data_layer._convert_timestamp_timezone(data_no_tz)
        post_convert_time = time.time() - start_time
        
        # Compare the results
        print("\nTimezone conversion results:")
        print(f"Database with timezone conversion: {db_query_time:.6f} seconds")
        print(f"Database without timezone conversion: {no_tz_query_time:.6f} seconds")
        print(f"Post-query timezone conversion: {post_convert_time:.6f} seconds")
        print(f"Total without DB conversion: {no_tz_query_time + post_convert_time:.6f} seconds")
        
        # Check if timezone is correctly set
        print("\nTimestamp timezone information:")
        
        if 'timestamp' in data_db.columns:
            db_timezone = data_db.schema['timestamp'].dtype.time_zone
            print(f"Database conversion timezone: {db_timezone}")
        else:
            print("No timestamp column found in database converted data")
        
        if 'timestamp' in data_post_convert.columns:
            post_timezone = data_post_convert.schema['timestamp'].dtype.time_zone
            print(f"Post-query conversion timezone: {post_timezone}")
        else:
            print("No timestamp column found in post-converted data")
            
        # Check if the values are the same
        print("\nData comparison:")
        if not data_db.is_empty() and not data_post_convert.is_empty():
            db_sample = data_db.head(1).to_dicts()[0]
            post_sample = data_post_convert.head(1).to_dicts()[0]
            
            print(f"DB timestamp: {db_sample['timestamp']}")
            print(f"Post timestamp: {post_sample['timestamp']}")
            
            # Check if timestamps are equal
            if db_sample['timestamp'] == post_sample['timestamp']:
                print("✅ Timestamps match exactly!")
            else:
                print("❌ Timestamps differ! Check implementation.")
        
        print(f"\n--- End of test for timezone: {tz} ---\n")

if __name__ == "__main__":
    test_timezone_conversion()
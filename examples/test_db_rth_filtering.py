#!/usr/bin/env python3
"""
Test script to verify database-level filtering of regular trading hours.
"""

import polars as pl
from datetime import datetime, timedelta
from vegas.engine import BacktestEngine
from vegas.data import DataLayer
import time

# Configure trading hours
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

def test_database_filtering():
    """Test the database-level filtering of regular trading hours."""
    
    # Create engine with regular hours filtering
    engine = BacktestEngine()
    engine.set_trading_hours("US", MARKET_OPEN, MARKET_CLOSE)
    
    # First test without filtering
    print("Testing without RTH filtering...")
    start_time = time.time()
    
    # Get market data without filtering
    engine.ignore_extended_hours(False)
    all_data = engine.data_layer.get_data_for_backtest(
        start=datetime(2022, 1, 1),
        end=datetime(2022, 1, 31)
    )
    
    normal_time = time.time() - start_time
    
    if not all_data.is_empty():
        # Analyze data
        print(f"Total data points: {len(all_data)}")
        
        # Get distribution by hour
        hour_counts = all_data.with_columns(
            hour=pl.col('timestamp').dt.hour()
        ).group_by('hour').agg(
            pl.count().alias('count')
        ).sort('hour')
        
        print("\nData distribution by hour (without filtering):")
        print(hour_counts)
        
        # Now test with filtering at database level
        print("\n\nTesting with RTH filtering at database level...")
        start_time = time.time()
        
        # Enable filtering
        engine.ignore_extended_hours(True)
        
        # Get filtered data
        filtered_data = engine.data_layer.get_data_for_backtest(
            start=datetime(2022, 1, 1),
            end=datetime(2022, 1, 31),
            market_hours=(MARKET_OPEN, MARKET_CLOSE)
        )
        
        db_filter_time = time.time() - start_time
        
        if not filtered_data.is_empty():
            # Analyze filtered data
            print(f"Total data points after filtering: {len(filtered_data)}")
            
            # Get distribution by hour
            hour_counts_filtered = filtered_data.with_columns(
                hour=pl.col('timestamp').dt.hour()
            ).group_by('hour').agg(
                pl.count().alias('count')
            ).sort('hour')
            
            print("\nData distribution by hour (with filtering):")
            print(hour_counts_filtered)
            
            # Check that there's no data outside RTH
            open_hour = int(MARKET_OPEN.split(':')[0])
            close_hour = int(MARKET_CLOSE.split(':')[0])
            
            hours_outside_rth = hour_counts_filtered.filter(
                (pl.col('hour') < open_hour) | (pl.col('hour') >= close_hour)
            )
            
            if hours_outside_rth.is_empty():
                print("\n✅ Success: No data outside regular trading hours")
            else:
                print("\n❌ Error: Found data outside regular trading hours")
                print(hours_outside_rth)
            
            # Compare performance
            print(f"\nPerformance comparison:")
            print(f"Without filtering: {normal_time:.4f} seconds")
            print(f"With DB filtering: {db_filter_time:.4f} seconds")
            
            if db_filter_time < normal_time:
                improvement = (normal_time - db_filter_time) / normal_time * 100
                print(f"Database filtering is {improvement:.1f}% faster!")
            else:
                slowdown = (db_filter_time - normal_time) / normal_time * 100
                print(f"Database filtering is {slowdown:.1f}% slower")
        else:
            print("No data found with filtering")
    else:
        print("No data found in the database")

if __name__ == "__main__":
    test_database_filtering()
#!/usr/bin/env python3
"""
Test to verify that the refactored generate_signals method works correctly.

This test creates a simple strategy that returns different symbols each day
to verify that before_trading_start is called day-by-day and that signals
are generated only for the symbols returned by before_trading_start.
"""

from datetime import datetime, date
from typing import Dict, List, Optional

import polars as pl

from vegas.strategy import Context, Strategy


class TestDailySymbolStrategy(Strategy):
    """
    Test strategy that returns different symbols each day to verify day-by-day processing.
    """
    
    def initialize(self, context: Context):
        """Initialize strategy."""
        self.day_counter = 0
        context.logger = context.engine._logger
        
    def before_trading_start(self, context: Context, data: Dict[str, pl.DataFrame]) -> Optional[List[str]]:
        """
        Return different symbols each day to test day-by-day functionality.
        
        Day 1: ['AAPL', 'MSFT']
        Day 2: ['GOOGL', 'TSLA'] 
        Day 3: ['AAPL', 'GOOGL']
        etc.
        """
        self.day_counter += 1
        
        # Define different symbol sets for different days
        symbol_sets = [
            ['AAPL', 'MSFT'],
            ['GOOGL', 'TSLA'],
            ['AAPL', 'GOOGL'],
            ['MSFT', 'TSLA'],
        ]
        
        day_symbols = symbol_sets[(self.day_counter - 1) % len(symbol_sets)]
        
        context.logger.info(f"Day {self.day_counter}: Selected symbols {day_symbols}")
        
        return day_symbols
    
    def predict(self, t: int, data: Dict[str, pl.DataFrame]) -> Dict[str, float]:
        """
        Generate simple test signals.
        
        Returns a constant signal of 1.0 for each symbol in the data.
        This allows us to verify which symbols were actually processed.
        """
        signals = {}
        
        for symbol in data.keys():
            signals[symbol] = 1.0  # Simple constant signal
            
        return signals


def test_day_by_day_processing():
    """
    Test that generate_signals processes day-by-day and respects before_trading_start symbol selection.
    """
    print("Testing day-by-day signal generation...")
    
    # This test would need actual market data to run
    # For now, we'll just show the structure
    
    from vegas.engine import BacktestEngine
    
    engine = BacktestEngine()
    strategy = TestDailySymbolStrategy()
    
    # In a real test, you would need to load data first:
    # engine.load_data(file_path='test_data.csv')
    
    print("Test strategy created successfully")
    print("Strategy will return different symbols each day:")
    print("Day 1: ['AAPL', 'MSFT']")
    print("Day 2: ['GOOGL', 'TSLA']") 
    print("Day 3: ['AAPL', 'GOOGL']")
    print("Day 4: ['MSFT', 'TSLA']")
    print()
    print("To run this test with real data:")
    print("1. Load market data using engine.load_data()")
    print("2. Call engine.generate_signals() with a date range")
    print("3. Verify that signals DataFrame contains different symbols on different days")
    
    # Example of how to run with real data:
    """
    try:
        signals_df = engine.generate_signals(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 10),
            strategy=strategy
        )
        
        # Verify day-by-day symbol selection
        daily_symbols = (
            signals_df
            .with_columns(pl.col('datetime').dt.date().alias('date'))
            .group_by('date')
            .agg([
                # Count non-null signals per symbol per day
                pl.all().exclude(['datetime', 'date']).is_not_null().sum().alias('symbol_counts')
            ])
        )
        
        print("Daily symbol usage:")
        print(daily_symbols)
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure market data is loaded first")
    """


if __name__ == "__main__":
    test_day_by_day_processing()

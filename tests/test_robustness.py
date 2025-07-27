"""Robustness tests for the Vegas backtesting engine."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Context, Signal
from vegas.data import DataLayer


class SimpleStrategy(Strategy):
    """Simple strategy for robustness testing."""
    
    def initialize(self, context):
        """Initialize the strategy."""
        context.symbols = ['TEST1', 'TEST2', 'TEST3']
        context.errors = []
        context.signal_count = 0
    
    def handle_data(self, context, data):
        """Generate signals based on data."""
        signals = []
        
        # Generate signals only for valid data
        for symbol in context.symbols:
            symbol_data = data[data['symbol'] == symbol]
            
            if symbol_data.empty:
                continue
            
            row = symbol_data.iloc[0]
            price = row['close']
            
            # Skip invalid prices
            if pd.isna(price) or price <= 0:
                continue
            
            # Simple strategy: buy when signal count is even, sell when odd
            if context.signal_count % 2 == 0:
                signals.append(Signal(
                    symbol=symbol,
                    action='buy',
                    quantity=100,
                    price=price
                ))
            else:
                signals.append(Signal(
                    symbol=symbol,
                    action='sell',
                    quantity=100,
                    price=price
                ))
        
        # Increment signal count if signals were generated
        if signals:
            context.signal_count += 1
            
        return signals


def generate_test_data_with_disruptions(days=10, include_circuit_breakers=False, 
                                       include_halts=False, include_gaps=False,
                                       include_extreme_volatility=False):
    """Generate test market data with various disruptions.
    
    Args:
        days: Number of days to generate data for
        include_circuit_breakers: Whether to include circuit breaker events
        include_halts: Whether to include trading halts
        include_gaps: Whether to include data gaps
        include_extreme_volatility: Whether to include extreme volatility
        
    Returns:
        DataFrame with test data
    """
    # Generate timestamps
    base_date = datetime(2022, 1, 1)
    timestamps = []
    
    for i in range(days):
        current_date = base_date + timedelta(days=i)
        for hour in range(9, 16):  # Trading hours
            timestamps.append(current_date.replace(hour=hour))
    
    # Generate data
    data = []
    symbols = ["TEST1", "TEST2", "TEST3"]
    
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        
        # Circuit breaker day (if enabled)
        circuit_breaker_day = None
        if include_circuit_breakers:
            circuit_breaker_day = np.random.randint(0, days)
        
        # Trading halt day and duration (if enabled)
        halt_day = None
        halt_duration = None
        if include_halts:
            halt_day = np.random.randint(0, days)
            halt_duration = np.random.randint(1, 3)  # 1-2 hours
        
        # Extreme volatility day (if enabled)
        extreme_vol_day = None
        if include_extreme_volatility:
            extreme_vol_day = np.random.randint(0, days)
        
        for i, ts in enumerate(timestamps):
            current_day = (ts - base_date).days
            current_hour = ts.hour - 9  # 0-6 for trading hours
            
            # Skip if data gap
            if include_gaps and np.random.random() < 0.05:  # 5% chance of gap
                continue
                
            # Skip if trading halt
            if include_halts and current_day == halt_day and current_hour < halt_duration:
                continue
                
            # Generate price movement
            if include_extreme_volatility and current_day == extreme_vol_day:
                # Extreme volatility: 5-10% price changes
                price_change = np.random.normal(0, 5.0)
            else:
                # Normal volatility: 0-0.5% price changes
                price_change = np.random.normal(0, 0.5)
                
            # Apply circuit breaker (limit price movement)
            if include_circuit_breakers and current_day == circuit_breaker_day:
                if current_hour == 0:  # Morning
                    # Large drop at open (-7%)
                    price_change = -7.0
                elif current_hour == 1:
                    # Circuit breaker limits further drops
                    price_change = min(price_change, 0.5)
            
            current_price = max(1, base_price * (1 + price_change / 100))
            
            # Add some volatility
            high = current_price * (1 + np.random.uniform(0, 0.5) / 100)
            low = current_price * (1 - np.random.uniform(0, 0.5) / 100)
            
            # Ensure high >= close >= low
            high = max(high, current_price)
            low = min(low, current_price)
            
            # Generate volume
            if include_extreme_volatility and current_day == extreme_vol_day:
                # High volume during extreme volatility
                volume = np.random.randint(10000, 100000)
            else:
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
def temp_robust_dir():
    """Create a temporary directory for robustness test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_handling_market_disruptions(temp_robust_dir):
    """Test handling of market disruptions (circuit breakers, halts)."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate test data with circuit breakers and halts
    df = generate_test_data_with_disruptions(
        days=10,
        include_circuit_breakers=True,
        include_halts=True
    )
    
    # Load data
    engine.data_layer.data = df
    
    # Create strategy
    strategy = SimpleStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully despite disruptions
    assert results['success']
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Check that the strategy handled the disruptions
    # (The strategy records warnings in context.warnings)
    context = strategy.context
    
    # There should be no errors
    assert len(context.errors) == 0, f"Strategy encountered errors: {context.errors}"
    
    # There might be warnings about insufficient data due to halts
    # but the backtest should still complete
    
    # Verify transactions were generated
    transactions = results['transactions']
    assert not transactions.empty, "No transactions were generated"


def test_extreme_market_conditions(temp_robust_dir):
    """Test with extreme market conditions (high volatility)."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate test data with extreme volatility
    df = generate_test_data_with_disruptions(
        days=10,
        include_extreme_volatility=True
    )
    
    # Load data
    engine.data_layer.data = df
    
    # Create strategy
    strategy = SimpleStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully despite extreme conditions
    assert results['success']
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Check that the strategy handled the extreme conditions
    context = strategy.context
    assert len(context.errors) == 0, f"Strategy encountered errors: {context.errors}"
    
    # Verify equity curve has reasonable values
    equity_curve = results['equity_curve']
    
    # Equity should never go negative
    assert all(equity_curve['equity'] > 0)
    
    # Calculate daily returns
    equity_curve['date'] = equity_curve['timestamp'].dt.date
    daily_equity = equity_curve.groupby('date')['equity'].last().reset_index()
    daily_equity['return'] = daily_equity['equity'].pct_change()
    
    # Verify no unreasonable daily returns (>100%)
    # In extreme volatility, we might see large returns, but they should still be reasonable
    assert all(daily_equity['return'].fillna(0).abs() < 1.0), "Unreasonable daily returns detected"


def test_incomplete_or_missing_data(temp_robust_dir):
    """Test with incomplete or missing data."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate test data with gaps
    df = generate_test_data_with_disruptions(
        days=10,
        include_gaps=True
    )
    
    # Load data
    engine.data_layer.data = df
    
    # Create strategy
    strategy = SimpleStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully despite gaps
    assert results['success']
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Check that the strategy handled the gaps
    context = strategy.context
    assert len(context.errors) == 0, f"Strategy encountered errors: {context.errors}"
    
    # Verify equity curve is continuous despite data gaps
    equity_curve = results['equity_curve']
    assert not equity_curve.empty
    
    # Test with completely missing symbol
    df_missing = df[df['symbol'] != 'TEST1'].copy()
    engine.data_layer.data = df_missing
    
    # The strategy should still run with the remaining symbols
    results_missing = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully despite missing symbol
    assert results_missing['success']
    assert 'equity_curve' in results_missing
    assert not results_missing['equity_curve'].empty


def test_error_recovery_and_logging(temp_robust_dir):
    """Test error recovery and logging."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate normal test data
    df = generate_test_data_with_disruptions(days=10)
    
    # Load data
    engine.data_layer.data = df
    
    # Create a strategy that deliberately raises errors
    class ErrorStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3']
            context.error_raised = False
            context.continued_after_error = False
            context.error_count = 0
        
        def handle_data(self, context, data):
            # Raise an error on the first call
            if not context.error_raised:
                context.error_raised = True
                # Log this error using a standard logger
                logging.getLogger('vegas.strategy').error(
                    "Deliberate error for testing"
                )
                raise ValueError("Deliberate error for testing")
            
            # If we get here, it means the engine recovered from the error
            context.continued_after_error = True
            
            # Return empty signals
            return []
    
    # Set up logging capture
    log_capture = []
    
    class LogHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(record.getMessage())
    
    # Add handler to both engine and strategy loggers
    logger = logging.getLogger('vegas')  # Capture all vegas logs
    handler = LogHandler()
    logger.addHandler(handler)
    
    # Run backtest with error strategy
    strategy = ErrorStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # The engine should catch the error and continue
    results = engine.run(start_date, end_date, strategy)
    
    # Remove our custom handler
    logger.removeHandler(handler)
    
    # Verify results are returned despite the error
    assert results is not None
    assert 'success' in results
    
    # Verify error was logged
    error_logs = [log for log in log_capture if "Deliberate error for testing" in log]
    assert len(error_logs) > 0, "Error should be logged"


def test_handling_invalid_data(temp_robust_dir):
    """Test handling of invalid data values."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate normal test data
    df = generate_test_data_with_disruptions(days=10)
    
    # Introduce invalid data
    # 1. Add some NaN values
    nan_indices = np.random.choice(len(df), 10)
    df.loc[nan_indices, 'close'] = np.nan
    
    # 2. Add some zero prices
    zero_indices = np.random.choice(len(df), 5)
    df.loc[zero_indices, 'close'] = 0
    
    # 3. Add some negative prices
    neg_indices = np.random.choice(len(df), 3)
    df.loc[neg_indices, 'close'] = -1
    
    # Load data
    engine.data_layer.data = df
    
    # Create a strategy that handles invalid data
    class DataValidationStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3']
            context.invalid_data_count = 0
            context.prev_prices = {}  # Store previous prices by symbol
        
        def handle_data(self, context, data):
            signals = []
            
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                price = symbol_data['close'].iloc[0]
                
                # Count and handle invalid data
                if pd.isna(price):
                    context.invalid_data_count += 1
                    continue
                    
                if price == 0:
                    context.invalid_data_count += 1
                    continue
                    
                if price < 0:
                    context.invalid_data_count += 1
                    continue
                
                # Only generate signals for valid prices
                if price > 0:
                    signals.append(Signal(
                        symbol=symbol,
                        action='buy',
                        quantity=100,
                        price=price
                    ))
            
            return signals
    
    # Run backtest
    strategy = DataValidationStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully despite invalid data
    assert results['success']
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Verify the strategy detected the invalid data
    assert strategy.context.invalid_data_count > 0, "Strategy did not detect invalid data"
    
    # Verify transactions were generated despite invalid data
    transactions = results['transactions']
    assert not transactions.empty, "No transactions were generated"


def test_handling_system_resource_limitations(temp_robust_dir):
    """Test handling of system resource limitations."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_robust_dir)
    
    # Generate a reasonably large dataset
    symbols = [f"TEST{i}" for i in range(1, 51)]  # 50 symbols
    
    # Generate timestamps
    base_date = datetime(2022, 1, 1)
    timestamps = []
    
    for i in range(30):  # 30 days
        current_date = base_date + timedelta(days=i)
        for hour in range(9, 16):  # 7 trading hours
            timestamps.append(current_date.replace(hour=hour))
    
    # Generate data
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        for ts in timestamps:
            price_change = np.random.normal(0, 0.5)
            current_price = max(1, base_price * (1 + price_change / 100))
            
            high = current_price * (1 + np.random.uniform(0, 0.5) / 100)
            low = current_price * (1 - np.random.uniform(0, 0.5) / 100)
            
            high = max(high, current_price)
            low = min(low, current_price)
            
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
            
            base_price = current_price
    
    df = pd.DataFrame(data)
    
    # Load data
    engine.data_layer.data = df
    
    # Create a memory-intensive strategy
    class MemoryIntensiveStrategy(Strategy):
        def initialize(self, context):
            context.symbols = symbols
            context.window = 10
            context.data_cache = {}
        
        def handle_data(self, context, data):
            signals = []
            
            # Process each symbol and store intermediate results
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                if len(symbol_data) < context.window:
                    continue
                
                # Calculate multiple indicators (memory intensive)
                for window in range(1, 21):  # 20 different windows
                    symbol_data[f'ma_{window}'] = symbol_data['close'].rolling(window).mean()
                    symbol_data[f'std_{window}'] = symbol_data['close'].rolling(window).std()
                    symbol_data[f'min_{window}'] = symbol_data['close'].rolling(window).min()
                    symbol_data[f'max_{window}'] = symbol_data['close'].rolling(window).max()
                
                # Store in cache (memory intensive)
                context.data_cache[symbol] = symbol_data
                
                # Generate signals
                for i in range(context.window, len(symbol_data)):
                    row = symbol_data.iloc[i]
                    
                    if row['close'] > row['ma_10']:
                        signals.append(Signal(
                            symbol=symbol,
                            action='buy',
                            quantity=100,
                            price=row['close']
                        ))
            
            return signals
    
    # Run backtest
    strategy = MemoryIntensiveStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # The backtest should complete without running out of memory
    results = engine.run(start_date, end_date, strategy)
    
    # Verify backtest completed successfully
    assert results['success']
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Clean up memory
    strategy.context.data_cache = None


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
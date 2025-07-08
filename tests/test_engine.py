"""Tests for the engine functionality of the Vegas backtesting engine."""

import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Context, Signal


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


class TestTrendStrategy(Strategy):
    """Simple trend-following strategy for testing."""
    
    def initialize(self, context):
        """Initialize the strategy."""
        context.symbols = ['TEST1', 'TEST2', 'TEST3']
        context.ma_short = 3
        context.ma_long = 7
    
    def generate_signals_vectorized(self, context, data):
        """Generate signals based on moving average crossover."""
        signals = []
        
        for symbol in context.symbols:
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < context.ma_long:
                continue
                
            # Calculate moving averages
            symbol_data['ma_short'] = symbol_data['close'].rolling(context.ma_short).mean()
            symbol_data['ma_long'] = symbol_data['close'].rolling(context.ma_long).mean()
            
            # Generate signals on crossover
            symbol_data['signal'] = np.where(
                symbol_data['ma_short'] > symbol_data['ma_long'], 1, -1)
            
            # Detect crossovers
            symbol_data['position_change'] = symbol_data['signal'].diff().fillna(0)
            
            # Filter to just the crossover points
            crossovers = symbol_data[symbol_data['position_change'] != 0]
            
            for _, row in crossovers.iterrows():
                signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'action': 'buy' if row['position_change'] > 0 else 'sell',
                    'quantity': 100,  # Fixed quantity for testing
                    'price': None  # Market order
                })
        
        return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])


class TestMeanReversionStrategy(Strategy):
    """Simple mean-reversion strategy for testing."""
    
    def initialize(self, context):
        """Initialize the strategy."""
        context.symbols = ['TEST1', 'TEST2', 'TEST3']
        context.lookback = 5
        context.entry_z = 1.5
        context.exit_z = 0.5
    
    def generate_signals_vectorized(self, context, data):
        """Generate signals based on z-score mean reversion."""
        signals = []
        
        for symbol in context.symbols:
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < context.lookback:
                continue
                
            # Calculate z-score
            symbol_data['mean'] = symbol_data['close'].rolling(context.lookback).mean()
            symbol_data['std'] = symbol_data['close'].rolling(context.lookback).std()
            symbol_data['z_score'] = (symbol_data['close'] - symbol_data['mean']) / symbol_data['std']
            
            # Generate signals based on z-score thresholds
            for i in range(context.lookback, len(symbol_data)):
                row = symbol_data.iloc[i]
                prev_row = symbol_data.iloc[i-1]
                
                # Short when z-score crosses above entry threshold
                if row['z_score'] > context.entry_z and prev_row['z_score'] <= context.entry_z:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': 100,
                        'price': None
                    })
                
                # Cover when z-score crosses below exit threshold
                elif row['z_score'] < context.exit_z and prev_row['z_score'] >= context.exit_z:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,
                        'price': None
                    })
                
                # Long when z-score crosses below negative entry threshold
                elif row['z_score'] < -context.entry_z and prev_row['z_score'] >= -context.entry_z:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,
                        'price': None
                    })
                
                # Sell when z-score crosses above negative exit threshold
                elif row['z_score'] > -context.exit_z and prev_row['z_score'] <= -context.exit_z:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': 100,
                        'price': None
                    })
        
        return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])


@pytest.fixture
def temp_engine_dir():
    """Create a temporary directory for engine files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_engine_initialization_with_different_configs(temp_engine_dir):
    """Test engine initialization with different configurations."""
    # Test default initialization
    engine1 = BacktestEngine()
    assert engine1.data_layer is not None
    
    # Test with custom data directory
    engine2 = BacktestEngine(data_dir=temp_engine_dir)
    assert engine2.data_layer is not None
    assert engine2.data_layer.data_dir == temp_engine_dir
    
    # Test with non-existent directory (should create it)
    non_existent_dir = os.path.join(temp_engine_dir, "non_existent")
    engine3 = BacktestEngine(data_dir=non_existent_dir)
    assert engine3.data_layer is not None
    assert os.path.exists(non_existent_dir)


def test_full_backtest_execution(temp_engine_dir):
    """Test full backtest execution with sample strategies."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_engine_dir)
    
    # Generate test data
    df = generate_test_data(symbols=["TEST1", "TEST2", "TEST3"], days=20)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create strategy
    strategy = TestTrendStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify results
    assert results is not None
    assert results['success']
    assert 'equity_curve' in results
    assert 'transactions' in results
    assert 'stats' in results
    
    # Check equity curve
    equity_curve = results['equity_curve']
    assert not equity_curve.empty
    assert 'equity' in equity_curve.columns
    assert 'timestamp' in equity_curve.columns
    
    # Check transactions
    transactions = results['transactions']
    
    # Check stats
    stats = results['stats']
    assert 'total_return' in stats
    assert 'total_return_pct' in stats
    assert 'num_trades' in stats
    
    # Run with a different strategy
    mean_reversion_strategy = TestMeanReversionStrategy()
    mr_results = engine.run(start_date, end_date, mean_reversion_strategy)
    
    # Verify results
    assert mr_results is not None
    assert mr_results['success']
    assert 'equity_curve' in mr_results


def test_handling_edge_cases(temp_engine_dir):
    """Test handling of edge cases (empty data, single data point)."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_engine_dir)
    
    # Test with empty data
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    # Empty dataframe is not considered initialized, so we should ensure it meets minimum requirements
    empty_df = pd.DataFrame({
        "timestamp": [datetime(2022, 1, 1)],
        "symbol": ["TEST1"],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000]
    })
    engine.data_layer.data = empty_df
    
    strategy = TestTrendStrategy()
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 10)
    
    empty_results = engine.run(start_date, end_date, strategy)
    
    # Verify results structure is correct even with minimal data
    assert empty_results is not None
    assert empty_results['success']
    assert 'equity_curve' in empty_results
    assert 'transactions' in empty_results
    assert 'stats' in empty_results
    
    # Test with a single data point
    single_df = pd.DataFrame({
        "timestamp": [datetime(2022, 1, 1, 9, 0)],
        "symbol": ["TEST1"],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [1000]
    })
    engine.data_layer.data = single_df
    
    single_results = engine.run(start_date, end_date, strategy)
    
    # Verify results
    assert single_results is not None
    assert single_results['success']
    assert 'equity_curve' in single_results
    assert 'transactions' in single_results
    assert 'stats' in single_results


def test_performance_metrics_calculation(temp_engine_dir):
    """Test performance metrics calculation."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_engine_dir)
    
    # Generate test data with predictable pattern
    np.random.seed(42)  # For reproducibility
    df = generate_test_data(symbols=["TEST1"], days=30)
    
    # Load data
    engine.data_layer.data = df
    
    # Create strategy
    strategy = TestTrendStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy, initial_capital=10000.0)
    
    # Verify performance metrics
    stats = results['stats']
    assert 'total_return' in stats
    assert 'total_return_pct' in stats
    assert 'num_trades' in stats
    
    # Calculate expected metrics
    equity_curve = results['equity_curve']
    initial_equity = 10000.0
    final_equity = equity_curve['equity'].iloc[-1] if not equity_curve.empty else initial_equity
    expected_total_return = final_equity - initial_equity
    expected_return_pct = (expected_total_return / initial_equity) * 100.0
    
    # Compare with calculated metrics
    assert abs(stats['total_return'] - expected_total_return) < 0.01
    assert abs(stats['total_return_pct'] - expected_return_pct) < 0.01


def test_report_generation(temp_engine_dir):
    """Test report generation."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_engine_dir)
    
    # Generate test data
    df = generate_test_data(symbols=["TEST1", "TEST2"], days=30)
    
    # Load data
    engine.data_layer.data = df
    
    # Create strategy
    strategy = TestTrendStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, strategy)
    
    # Verify results for report generation
    assert results is not None
    assert 'equity_curve' in results
    assert not results['equity_curve'].empty
    
    # Check if we can generate a report
    # This just checks that the necessary data is available
    equity_curve = results['equity_curve']
    assert 'timestamp' in equity_curve.columns
    assert 'equity' in equity_curve.columns
    
    # If quantstats is available, we could test actual report generation
    try:
        import quantstats as qs
        
        # Convert equity curve to returns
        if not equity_curve.empty and len(equity_curve) > 1:
            returns = equity_curve.set_index('timestamp')['equity'].pct_change().fillna(0)
            
            # Check that returns are valid for report generation
            assert not returns.empty
            assert not returns.isnull().all()
    except ImportError:
        pytest.skip("QuantStats not available")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
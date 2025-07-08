"""Tests for preventing common biases in the Vegas backtesting engine."""

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
from vegas.data import DataLayer


class LookaheadStrategy(Strategy):
    """Strategy that attempts to use future data (should be prevented)."""
    
    def initialize(self, context):
        """Initialize the strategy."""
        context.symbols = ['TEST1']
        context.lookahead_periods = 1  # Try to look ahead by 1 period
    
    def generate_signals_vectorized(self, context, data):
        """Attempt to generate signals using future data."""
        signals = []
        
        for symbol in context.symbols:
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < 2:
                continue
                
            # Attempt to use future prices by shifting
            symbol_data['next_close'] = symbol_data['close'].shift(-1)
            
            # Generate "perfect" signals based on future knowledge
            for i in range(len(symbol_data) - 1):  # Skip last row (NaN)
                row = symbol_data.iloc[i]
                
                if pd.isna(row['next_close']):
                    continue
                    
                # Perfect strategy: buy if price will go up, sell if it will go down
                if row['next_close'] > row['close']:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,
                        'price': None
                    })
                else:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': 100,
                        'price': None
                    })
        
        return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])


class PointInTimeStrategy(Strategy):
    """Strategy that properly uses point-in-time data."""
    
    def initialize(self, context):
        """Initialize the strategy."""
        context.symbols = ['TEST1']
        context.ma_length = 3
    
    def generate_signals_vectorized(self, context, data):
        """Generate signals using only past data at each point in time."""
        signals = []
        
        for symbol in context.symbols:
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < context.ma_length:
                continue
                
            # Calculate moving average (uses only past data)
            symbol_data['ma'] = symbol_data['close'].rolling(context.ma_length).mean()
            
            # Generate signals based on price vs MA
            for i in range(context.ma_length, len(symbol_data)):
                row = symbol_data.iloc[i]
                
                if pd.isna(row['ma']):
                    continue
                    
                # Buy if price is above MA, sell if below
                if row['close'] > row['ma']:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,
                        'price': None
                    })
                else:
                    signals.append({
                        'timestamp': row['timestamp'],
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': 100,
                        'price': None
                    })
        
        return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])


def generate_test_data(days=10, with_delisted=False, with_gaps=False, with_backfill=False):
    """Generate test market data with options for testing different biases.
    
    Args:
        days: Number of days to generate data for
        with_delisted: Whether to include delisted securities
        with_gaps: Whether to include data gaps
        with_backfill: Whether to include backfilled data
        
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
    
    # Regular symbols
    symbols = ["TEST1", "TEST2", "TEST3"]
    
    # Add delisted symbol if requested
    if with_delisted:
        symbols.append("DELISTED")
    
    for symbol in symbols:
        base_price = np.random.uniform(50, 200)
        
        # Determine when the symbol should be delisted
        delisting_point = None
        if with_delisted and symbol == "DELISTED":
            delisting_point = len(timestamps) // 2  # Delist halfway through
        
        for i, ts in enumerate(timestamps):
            # Skip if this symbol is delisted
            if delisting_point is not None and i >= delisting_point and symbol == "DELISTED":
                continue
                
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
    
    df = pd.DataFrame(data)
    
    # Add backfilled data if requested
    if with_backfill:
        # Create a new symbol with backfilled data
        backfill_symbol = "BACKFILLED"
        backfill_data = []
        
        # Use the last price and backfill it
        last_price = np.random.uniform(50, 200)
        
        for ts in timestamps:
            backfill_data.append({
                "timestamp": ts,
                "symbol": backfill_symbol,
                "open": last_price,
                "high": last_price,
                "low": last_price,
                "close": last_price,
                "volume": 1000
            })
        
        backfill_df = pd.DataFrame(backfill_data)
        df = pd.concat([df, backfill_df], ignore_index=True)
    
    return df


@pytest.fixture
def temp_bias_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_no_lookahead_bias(temp_bias_dir):
    """Test that the engine prevents lookahead bias."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data with clear pattern
    df = generate_test_data(days=10)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create lookahead strategy
    lookahead_strategy = LookaheadStrategy()
    
    # Run backtest
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    results = engine.run(start_date, end_date, lookahead_strategy)
    
    # Verify results
    assert results is not None
    assert results['success']
    
    # Check performance - should not be unrealistically good
    # A perfect strategy using future data would have nearly 100% win rate
    # and very high returns
    stats = results['stats']
    
    # The strategy should not be able to achieve unrealistic returns
    # due to lookahead prevention
    transactions = results['transactions']
    
    if not transactions.empty:
        # Calculate win rate if there are transactions
        buy_txns = transactions[transactions['quantity'] > 0]
        sell_txns = transactions[transactions['quantity'] < 0]
        
        # If we have both buys and sells, we can check profitability
        if not buy_txns.empty and not sell_txns.empty:
            # A perfect strategy would have close to 100% profitable trades
            # Our implementation should prevent this
            assert stats['total_return_pct'] < 90.0, "Strategy appears to have lookahead bias"
    
    # Compare with a proper point-in-time strategy
    pt_strategy = PointInTimeStrategy()
    pt_results = engine.run(start_date, end_date, pt_strategy)
    
    # Both strategies should produce similar results if lookahead is properly prevented
    lookahead_return = results['stats']['total_return_pct']
    pt_return = pt_results['stats']['total_return_pct']
    
    # The lookahead strategy should not dramatically outperform the point-in-time strategy
    # Allow for some variance but not orders of magnitude difference
    assert abs(lookahead_return - pt_return) < 50.0, "Lookahead strategy significantly outperforms point-in-time strategy"


def test_proper_data_windowing(temp_bias_dir):
    """Test proper data windowing in vectorized operations."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data
    df = generate_test_data(days=10)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that uses windowed operations
    class WindowStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1']
            context.window = 5
        
        def generate_signals_vectorized(self, context, data):
            signals = []
            
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < context.window:
                    continue
                
                # Calculate rolling mean and std
                symbol_data['mean'] = symbol_data['close'].rolling(context.window).mean()
                symbol_data['std'] = symbol_data['close'].rolling(context.window).std()
                
                # These operations should only use past data
                # First window-1 rows should have NaN values
                for i in range(context.window - 1):
                    assert pd.isna(symbol_data['mean'].iloc[i])
                    assert pd.isna(symbol_data['std'].iloc[i])
                
                # Window row and beyond should have values
                for i in range(context.window - 1, len(symbol_data)):
                    assert not pd.isna(symbol_data['mean'].iloc[i])
                    assert not pd.isna(symbol_data['std'].iloc[i])
                
                # Generate signals (not important for this test)
                for i in range(context.window, len(symbol_data)):
                    signals.append({
                        'timestamp': symbol_data.iloc[i]['timestamp'],
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': 100,
                        'price': None
                    })
            
            return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
    
    # Run backtest
    strategy = WindowStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if windowing is correct
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_point_in_time_data_access(temp_bias_dir):
    """Test point-in-time data access patterns."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data
    df = generate_test_data(days=10)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that verifies point-in-time data access
    class PointInTimeAccessStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1']
            context.current_index = 0
            context.data_points = []
        
        def generate_signals_vectorized(self, context, data):
            # This strategy doesn't generate signals
            # It just verifies that data access follows point-in-time rules
            
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
                
                # Store all timestamps and prices for verification
                for _, row in symbol_data.iterrows():
                    context.data_points.append({
                        'timestamp': row['timestamp'],
                        'close': row['close']
                    })
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that data points are in chronological order
            timestamps = [dp['timestamp'] for dp in context.data_points]
            assert timestamps == sorted(timestamps), "Data points not in chronological order"
    
    # Run backtest
    strategy = PointInTimeAccessStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if point-in-time access is correct
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_with_delisted_securities(temp_bias_dir):
    """Test with datasets containing delisted securities."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data with a delisted security
    df = generate_test_data(days=10, with_delisted=True)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that handles delisted securities
    class DelistingStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3', 'DELISTED']
            context.delisted_detected = False
        
        def generate_signals_vectorized(self, context, data):
            signals = []
            
            # Check for delisted symbol
            symbols_in_data = data['symbol'].unique()
            
            # DELISTED should be in the data but not for all timestamps
            assert 'DELISTED' in symbols_in_data, "Delisted symbol missing from data"
            
            delisted_data = data[data['symbol'] == 'DELISTED']
            all_data = data[data['symbol'] == 'TEST1']
            
            # Verify that delisted data has fewer timestamps
            assert len(delisted_data) < len(all_data), "Delisted security not properly handled"
            context.delisted_detected = True
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that delisting was detected
            assert context.delisted_detected, "Delisting was not detected"
    
    # Run backtest
    strategy = DelistingStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if delisted securities are handled correctly
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_universe_selection_at_specific_points(temp_bias_dir):
    """Test universe selection at specific points in time."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data with a delisted security
    df = generate_test_data(days=10, with_delisted=True)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that checks universe at different points in time
    class UniverseStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3', 'DELISTED']
            context.universe_checks = {}
        
        def generate_signals_vectorized(self, context, data):
            # Get all timestamps
            timestamps = sorted(data['timestamp'].unique())
            
            # Check universe at different points in time
            early_point = timestamps[0]
            mid_point = timestamps[len(timestamps) // 2]
            late_point = timestamps[-1]
            
            # Get universe at each point
            early_universe = set(data[data['timestamp'] == early_point]['symbol'].unique())
            mid_universe = set(data[data['timestamp'] == mid_point]['symbol'].unique())
            late_universe = set(data[data['timestamp'] == late_point]['symbol'].unique())
            
            # Store for verification
            context.universe_checks = {
                'early': early_universe,
                'mid': mid_universe,
                'late': late_universe
            }
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that universe changes over time
            early = context.universe_checks['early']
            mid = context.universe_checks['mid']
            late = context.universe_checks['late']
            
            # Early universe should contain all symbols
            assert 'DELISTED' in early, "Delisted symbol missing from early universe"
            assert len(early) == 4, "Early universe missing symbols"
            
            # Late universe should not contain delisted symbol
            assert 'DELISTED' not in late, "Delisted symbol incorrectly present in late universe"
            assert len(late) == 3, "Late universe has incorrect number of symbols"
    
    # Run backtest
    strategy = UniverseStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if universe selection is correct
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_with_backfilled_data(temp_bias_dir):
    """Test with datasets containing backfilled data."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data with backfilled data
    df = generate_test_data(days=10, with_backfill=True)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that detects backfilled data
    class BackfillDetectionStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3', 'BACKFILLED']
            context.backfill_detected = False
        
        def generate_signals_vectorized(self, context, data):
            # Check for backfilled symbol
            backfilled_data = data[data['symbol'] == 'BACKFILLED']
            
            if not backfilled_data.empty:
                # Backfilled data should have constant prices
                price_variance = backfilled_data['close'].var()
                
                # Variance should be very low or zero for backfilled data
                if price_variance < 0.0001:
                    context.backfill_detected = True
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that backfilled data was detected
            assert context.backfill_detected, "Backfilled data was not detected"
    
    # Run backtest
    strategy = BackfillDetectionStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if backfilled data is detected
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_detection_and_handling_of_data_gaps(temp_bias_dir):
    """Test detection and handling of data gaps."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data with gaps
    df = generate_test_data(days=10, with_gaps=True)
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that detects and handles data gaps
    class GapDetectionStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3']
            context.gaps_detected = False
        
        def generate_signals_vectorized(self, context, data):
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 2:
                    continue
                
                # Calculate time differences between consecutive rows
                symbol_data['time_diff'] = symbol_data['timestamp'].diff()
                
                # Check for gaps (time differences larger than expected)
                # For hourly data, normal diff is 1 hour
                expected_diff = pd.Timedelta(hours=1)
                gaps = symbol_data[symbol_data['time_diff'] > expected_diff]
                
                if not gaps.empty:
                    context.gaps_detected = True
                    break
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that gaps were detected
            assert context.gaps_detected, "Data gaps were not detected"
    
    # Run backtest
    strategy = GapDetectionStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if data gaps are detected
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


def test_data_quality_validation(temp_bias_dir):
    """Test data quality validation."""
    # Initialize engine
    engine = BacktestEngine(data_dir=temp_bias_dir)
    
    # Generate test data
    df = generate_test_data(days=10)
    
    # Introduce some data quality issues
    # 1. Add some outliers
    outlier_idx = np.random.choice(len(df), 5)
    df.loc[outlier_idx, 'close'] = df.loc[outlier_idx, 'close'] * 10
    
    # 2. Add some zero prices
    zero_idx = np.random.choice(len(df), 3)
    df.loc[zero_idx, 'close'] = 0
    
    # 3. Add some negative prices (invalid)
    neg_idx = np.random.choice(len(df), 2)
    df.loc[neg_idx, 'close'] = -1
    
    # Load data directly into the data layer
    engine.data_layer.data = df
    
    # Create a strategy that validates data quality
    class DataQualityStrategy(Strategy):
        def initialize(self, context):
            context.symbols = ['TEST1', 'TEST2', 'TEST3']
            context.quality_issues = {
                'outliers': 0,
                'zeros': 0,
                'negatives': 0
            }
        
        def generate_signals_vectorized(self, context, data):
            for symbol in context.symbols:
                symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
                
                if len(symbol_data) < 2:
                    continue
                
                # Calculate returns
                symbol_data['return'] = symbol_data['close'].pct_change()
                
                # Detect outliers (returns > 50%)
                outliers = symbol_data[abs(symbol_data['return']) > 0.5]
                context.quality_issues['outliers'] += len(outliers)
                
                # Detect zeros
                zeros = symbol_data[symbol_data['close'] == 0]
                context.quality_issues['zeros'] += len(zeros)
                
                # Detect negatives
                negatives = symbol_data[symbol_data['close'] < 0]
                context.quality_issues['negatives'] += len(negatives)
            
            # Return empty signals
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        def analyze(self, context, results):
            # Verify that data quality issues were detected
            assert context.quality_issues['outliers'] > 0, "Outliers not detected"
            assert context.quality_issues['zeros'] > 0, "Zeros not detected"
            assert context.quality_issues['negatives'] > 0, "Negatives not detected"
    
    # Run backtest
    strategy = DataQualityStrategy()
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    
    # This should run without assertion errors if data quality issues are detected
    results = engine.run(start_date, end_date, strategy)
    assert results['success']


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
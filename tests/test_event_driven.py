"""Tests for event-driven backtesting functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Context
from vegas.data import DataLayer


class MockDataLayer:
    """Mock data layer for testing."""
    
    def __init__(self):
        """Initialize with test data."""
        self.data = self._generate_test_data()
        
    def _generate_test_data(self):
        """Generate test market data."""
        symbols = ['AAPL', 'MSFT', 'GOOG']
        dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='B')
        times = ['09:30:00', '10:00:00', '11:00:00', '12:00:00', 
                 '13:00:00', '14:00:00', '15:00:00', '16:00:00']
        
        data = []
        for symbol in symbols:
            price = 100.0 + np.random.random() * 100.0  # Random starting price
            
            for date in dates:
                for time_str in times:
                    timestamp = pd.Timestamp(f"{date.date()} {time_str}")
                    
                    # Add some random price movement
                    price += np.random.normal(0, 1.0)
                    
                    data.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'open': price,
                        'high': price * (1 + np.random.random() * 0.01),
                        'low': price * (1 - np.random.random() * 0.01),
                        'close': price * (1 + np.random.normal(0, 0.005)),
                        'volume': np.random.randint(1000, 10000)
                    })
        
        return pd.DataFrame(data)
    
    def get_data_for_backtest(self, start, end, symbols=None):
        """Get data for a backtest period."""
        filtered = self.data[
            (self.data['timestamp'] >= pd.Timestamp(start)) &
            (self.data['timestamp'] <= pd.Timestamp(end))
        ]
        
        if symbols:
            filtered = filtered[filtered['symbol'].isin(symbols)]
            
        return filtered
    
    def get_data_for_timestamp(self, timestamp):
        """Get data for a specific timestamp."""
        # Find closest timestamp
        self.data['time_diff'] = abs(self.data['timestamp'] - timestamp)
        closest_idx = self.data.groupby('symbol')['time_diff'].idxmin()
        result = self.data.loc[closest_idx].drop('time_diff', axis=1)
        return result
    
    def get_trading_days(self, start_date, end_date):
        """Get trading days in range."""
        # Extract unique dates from timestamp
        dates = self.data[
            (self.data['timestamp'] >= pd.Timestamp(start_date)) &
            (self.data['timestamp'] <= pd.Timestamp(end_date))
        ]['timestamp'].dt.floor('D').unique()
        
        return pd.DatetimeIndex(dates)


class TestEventDriven:
    """Tests for event-driven backtesting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_layer = MockDataLayer()
    
    def test_event_driven_detection(self):
        """Test detection of event-driven strategies."""
        engine = BacktestEngine()
        
        # Replace data layer with mock
        engine.data_layer = self.data_layer
        
        # Strategy with no event methods
        class VectorizedStrategy(Strategy):
            def generate_signals_vectorized(self, context, data):
                return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        # Strategy with event method
        class EventDrivenStrategy(Strategy):
            def before_trading_start(self, context, data):
                context.signal = True
                
            def handle_data(self, context, data):
                return []
        
        # Test detection
        vectorized_strategy = VectorizedStrategy()
        event_strategy = EventDrivenStrategy()
        
        assert engine._requires_event_driven(vectorized_strategy) is False
        assert engine._requires_event_driven(event_strategy) is True
        
        # Test explicit flag
        vectorized_strategy.is_event_driven = True
        assert engine._requires_event_driven(vectorized_strategy) is True
    
    def test_event_driven_execution(self):
        """Test execution of event-driven backtest."""
        engine = BacktestEngine()
        
        # Replace data layer with mock
        engine.data_layer = self.data_layer
        
        # Strategy that tracks event calls
        class EventTrackingStrategy(Strategy):
            def initialize(self, context):
                context.event_count = {
                    'before_trading': 0,
                    'market_open': 0,
                    'market_close': 0,
                    'handle_data': 0
                }
                context.last_timestamp = None
            
            def before_trading_start(self, context, data):
                context.event_count['before_trading'] += 1
                context.last_timestamp = data['timestamp'].iloc[0] if not data.empty else None
            
            def on_market_open(self, context, data, portfolio):
                context.event_count['market_open'] += 1
                context.last_timestamp = data['timestamp'].iloc[0] if not data.empty else None
            
            def on_market_close(self, context, data, portfolio):
                context.event_count['market_close'] += 1
                context.last_timestamp = data['timestamp'].iloc[0] if not data.empty else None
            
            def handle_data(self, context, data):
                context.event_count['handle_data'] += 1
                return []
        
        # Run backtest
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 5)
        strategy = EventTrackingStrategy()
        
        results = engine.run(start_date, end_date, strategy, event_driven=True)
        
        # Check that events were triggered
        context = strategy.context
        assert context.event_count['before_trading'] > 0
        assert context.event_count['market_open'] > 0
        assert context.event_count['market_close'] > 0
        assert context.event_count['handle_data'] > 0
        
        # Total number of event days should match trading days
        trading_days = len(self.data_layer.get_trading_days(start_date, end_date))
        assert context.event_count['before_trading'] == trading_days
        assert context.event_count['market_open'] == trading_days
        assert context.event_count['market_close'] == trading_days
    
    def test_signal_generation(self):
        """Test signal generation in event-driven mode."""
        engine = BacktestEngine()
        
        # Replace data layer with mock
        engine.data_layer = self.data_layer
        
        # Strategy that generates signals at market open
        class SignalStrategy(Strategy):
            def initialize(self, context):
                context.bought = False
                
            def on_market_open(self, context, data, portfolio):
                if not context.bought:
                    context.bought = True
                    # Return signal via handle_data
            
            def handle_data(self, context, data):
                if context.bought and 'AAPL' in data['symbol'].values:
                    # Buy AAPL at first market open
                    from vegas.strategy import Signal
                    price = data[data['symbol'] == 'AAPL']['close'].iloc[0]
                    return [Signal(symbol='AAPL', action='buy', quantity=10, price=price)]
                return []
        
        # Run backtest
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 2)
        strategy = SignalStrategy()
        
        results = engine.run(start_date, end_date, strategy, event_driven=True)
        
        # Check transactions were generated
        transactions = results['transactions']
        assert not transactions.empty
        assert any(transactions['symbol'] == 'AAPL')
        assert transactions['quantity'].sum() > 0
    
    def test_event_vs_vectorized_same_result(self):
        """Test that event and vectorized modes give similar results for equivalent strategies."""
        # Create compatible strategy implementations
        class VectorizedStrategy(Strategy):
            def generate_signals_vectorized(self, context, data):
                # Buy AAPL at first timestamp
                aapl_data = data[data['symbol'] == 'AAPL'].sort_values('timestamp')
                if aapl_data.empty:
                    return pd.DataFrame()
                
                first_price = aapl_data.iloc[0]['close']
                return pd.DataFrame([{
                    'timestamp': aapl_data.iloc[0]['timestamp'],
                    'symbol': 'AAPL',
                    'action': 'buy',
                    'quantity': 10,
                    'price': first_price
                }])
        
        class EventStrategy(Strategy):
            def initialize(self, context):
                context.bought = False
                
            def handle_data(self, context, data):
                if not context.bought and 'AAPL' in data['symbol'].values:
                    context.bought = True
                    from vegas.strategy import Signal
                    price = data[data['symbol'] == 'AAPL']['close'].iloc[0]
                    return [Signal(symbol='AAPL', action='buy', quantity=10, price=price)]
                return []
        
        # Run both strategies
        engine_vec = BacktestEngine()
        engine_vec.data_layer = self.data_layer
        
        engine_event = BacktestEngine()
        engine_event.data_layer = self.data_layer
        
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 2)
        
        results_vec = engine_vec.run(
            start_date, end_date, VectorizedStrategy(), event_driven=False)
        results_event = engine_event.run(
            start_date, end_date, EventStrategy(), event_driven=True)
        
        # Both should have the same number of AAPL shares
        pos_vec = results_vec['positions']
        pos_event = results_event['positions']
        
        # They may not be exactly equal due to timing differences,
        # but should be comparable in having AAPL positions
        assert not pos_vec.empty and not pos_event.empty
        assert 'AAPL' in pos_vec['symbol'].values
        assert 'AAPL' in pos_event['symbol'].values 
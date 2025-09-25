"""Tests for the on_trade callback functionality."""

from datetime import datetime
from unittest.mock import MagicMock
import polars as pl

from vegas.strategy import Strategy, Signal, Context
from vegas.engine import BacktestEngine
from vegas.data import DataLayer


class MockOnTradeStrategy(Strategy):
    """Mock strategy for testing on_trade callback."""
    
    def __init__(self):
        super().__init__()
        self.on_trade_calls = []
        
    def initialize(self, context: Context) -> None:
        context.symbol = "TEST"
        
    def handle_data(self, context: Context, data: pl.DataFrame) -> list[Signal]:
        """Generate a simple buy signal."""
        return [Signal(symbol="TEST", quantity=100, order_type="market")]
        
    def on_trade(self, context: Context, trade_event: dict, portfolio) -> None:
        """Track calls to on_trade."""
        self.on_trade_calls.append(trade_event)


def test_on_trade_callback_is_called():
    """Test that on_trade is called when trades are executed."""
    # Create test data with manual timestamps
    base_time = datetime(2023, 1, 1, 9, 30)
    timestamps = [base_time]
    for i in range(1, 8):  # Create hourly timestamps
        timestamps.append(datetime(2023, 1, 1, 9 + i, 30))
    
    test_data = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["TEST"] * len(timestamps),
        "open": [100.0] * len(timestamps),
        "high": [101.0] * len(timestamps),
        "low": [99.0] * len(timestamps),
        "close": [100.5] * len(timestamps),
        "volume": [1000] * len(timestamps),
    })
    
    timestamps_series = pl.Series("timestamp", timestamps)
    
    # Create mock data layer
    data_layer = MagicMock(spec=DataLayer)
    data_layer.get_data_for_backtest.return_value = test_data
    data_layer.get_unified_timestamp_index.return_value = timestamps_series
    data_layer.timezone = "UTC"
    
    # Create engine and strategy
    engine = BacktestEngine()
    engine.data_layer = data_layer
    engine.data_portal = DataPortal(data_layer)
    
    strategy = MockOnTradeStrategy()
    
    # Run backtest
    results = engine.run(
        start=datetime(2023, 1, 1, 9, 30),
        end=datetime(2023, 1, 1, 16, 0),
        strategy=strategy,
        initial_capital=100000.0,
        frequency="1h"
    )
    
    # Verify that on_trade was called
    assert len(strategy.on_trade_calls) > 0, "on_trade should have been called"
    
    # Check the structure of trade_event
    trade_event = strategy.on_trade_calls[0]
    required_keys = [
        "timestamp", "transaction_id", "order_id", "symbol", 
        "quantity", "price", "commission", "value"
    ]
    
    for key in required_keys:
        assert key in trade_event, f"trade_event should contain '{key}'"
    
    # Verify trade event values
    assert trade_event["symbol"] == "TEST"
    assert trade_event["quantity"] > 0  # Should be a buy
    assert isinstance(trade_event["timestamp"], datetime)
    assert trade_event["price"] > 0
    assert trade_event["value"] != 0


def test_on_trade_callback_error_handling():
    """Test that errors in on_trade don't break the backtest."""
    
    class ErrorStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.trade_executed = False
            
        def initialize(self, context: Context) -> None:
            context.symbol = "TEST"
            
        def handle_data(self, context: Context, data: pl.DataFrame) -> list[Signal]:
            if not self.trade_executed:
                self.trade_executed = True
                return [Signal(symbol="TEST", quantity=100, order_type="market")]
            return []
            
        def on_trade(self, context: Context, trade_event: dict, portfolio) -> None:
            """Intentionally raise an error."""
            raise ValueError("Test error in on_trade")
    
    # Create test data with manual timestamps
    timestamps = [datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 10, 30)]
    
    test_data = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["TEST"] * len(timestamps),
        "open": [100.0] * len(timestamps),
        "high": [101.0] * len(timestamps),
        "low": [99.0] * len(timestamps),
        "close": [100.5] * len(timestamps),
        "volume": [1000] * len(timestamps),
    })
    
    timestamps_series = pl.Series("timestamp", timestamps)
    
    # Create mock data layer
    data_layer = MagicMock(spec=DataLayer)
    data_layer.get_data_for_backtest.return_value = test_data
    data_layer.get_unified_timestamp_index.return_value = timestamps_series
    data_layer.timezone = "UTC"
    
    # Create engine and strategy
    engine = BacktestEngine()
    engine.data_layer = data_layer
    engine._initialize_data_portal()
    
    strategy = ErrorStrategy()
    
    # Run backtest - should not raise an exception
    results = engine.run(
        start=datetime(2023, 1, 1, 9, 30),
        end=datetime(2023, 1, 1, 10, 30),
        strategy=strategy,
        initial_capital=100000.0,
        frequency="1h"
    )
    
    # Backtest should complete despite the error
    assert results is not None
    assert "stats" in results.stats


def test_on_trade_not_called_when_no_trades():
    """Test that on_trade is not called when no trades are executed."""
    
    class NoTradeStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.on_trade_calls = []
            
        def initialize(self, context: Context) -> None:
            pass
            
        def handle_data(self, context: Context, data: pl.DataFrame) -> list[Signal]:
            # Never generate any signals
            return []
            
        def on_trade(self, context: Context, trade_event: dict, portfolio) -> None:
            self.on_trade_calls.append(trade_event)
    
    # Create test data with manual timestamps
    timestamps = [datetime(2023, 1, 1, 9, 30), datetime(2023, 1, 1, 10, 30)]
    
    test_data = pl.DataFrame({
        "timestamp": timestamps,
        "symbol": ["TEST"] * len(timestamps),
        "open": [100.0] * len(timestamps),
        "high": [101.0] * len(timestamps),
        "low": [99.0] * len(timestamps),
        "close": [100.5] * len(timestamps),
        "volume": [1000] * len(timestamps),
    })
    
    timestamps_series = pl.Series("timestamp", timestamps)
    
    # Create mock data layer
    data_layer = MagicMock(spec=DataLayer)
    data_layer.get_data_for_backtest.return_value = test_data
    data_layer.get_unified_timestamp_index.return_value = timestamps_series
    data_layer.timezone = "UTC"
    
    # Create engine and strategy
    engine = BacktestEngine()
    engine.data_layer = data_layer
    engine._initialize_data_portal()
    
    strategy = NoTradeStrategy()
    
    # Run backtest
    results = engine.run(
        start=datetime(2023, 1, 1, 9, 30),
        end=datetime(2023, 1, 1, 10, 30),
        strategy=strategy,
        initial_capital=100000.0,
        frequency="1h"
    )
    
    # Verify that on_trade was not called
    assert len(strategy.on_trade_calls) == 0, "on_trade should not have been called when no trades executed"


def run_tests():
    """Run all tests manually."""
    print("Running on_trade callback tests...")
    
    try:
        test_on_trade_callback_is_called()
        print("✓ test_on_trade_callback_is_called passed")
    except Exception as e:
        print(f"✗ test_on_trade_callback_is_called failed: {e}")
    
    try:
        test_on_trade_callback_error_handling()
        print("✓ test_on_trade_callback_error_handling passed")
    except Exception as e:
        print(f"✗ test_on_trade_callback_error_handling failed: {e}")
    
    try:
        test_on_trade_not_called_when_no_trades()
        print("✓ test_on_trade_not_called_when_no_trades passed")
    except Exception as e:
        print(f"✗ test_on_trade_not_called_when_no_trades failed: {e}")
        
    print("Tests completed!")


if __name__ == "__main__":
    run_tests()

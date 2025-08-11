"""Tests for the Strategy class (Polars-only, API-aligned)."""

from datetime import datetime

import polars as pl

from vegas.strategy import Context, Signal, Strategy


class TestStrategy(Strategy):
    """Simple strategy for testing."""

    def initialize(self, context):
        """Initialize test strategy."""
        context.test_param = 10

    def handle_data(self, context, data):
        """Generate test signals."""
        signals = []

        # Generate a signal for the first symbol in the data
        if data:
            symbol = next(iter(data.keys()))
            # vegas Signal API uses quantity sign to encode side; omit 'action'
            signals.append(Signal(symbol=symbol, quantity=100))

        return signals


def test_strategy_initialization():
    """Test strategy initialization."""
    strategy = TestStrategy()
    context = Context()
    strategy.initialize(context)

    assert context.test_param == 10


def test_signal_generation():
    """Test signal generation with Polars input."""
    strategy = TestStrategy()
    context = Context()
    strategy.initialize(context)

    # Create test data (Polars)
    data = {
        "AAPL": pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [150.0],
                "high": [155.0],
                "low": [149.0],
                "close": [153.0],
                "volume": [1000000],
            }
        )
    }

    signals = strategy.handle_data(context, data)

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].quantity == 100


def test_before_trading_start():
    """Test before_trading_start method."""
    strategy = TestStrategy()
    context = Context()
    strategy.initialize(context)

    # Create test data (Polars)
    data = {
        "AAPL": pl.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [150.0],
                "high": [155.0],
                "low": [149.0],
                "close": [153.0],
                "volume": [1000000],
            }
        )
    }

    # Method should not raise an exception
    strategy.before_trading_start(context, data)

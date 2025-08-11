"""Pytest configuration for Vegas tests."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from vegas.broker import Broker
from vegas.portfolio import Portfolio
from vegas.strategy import Context, Signal, Strategy


@pytest.fixture
def sample_data():
    """Create sample market data."""
    dates = pd.date_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), freq="1H"
    )

    symbols = ["AAPL", "MSFT", "GOOG"]
    data = {}

    for symbol in symbols:
        # Create synthetic price data
        base_price = 100 + (ord(symbol[0]) % 10) * 10  # Different base price per symbol
        n = len(dates)

        # Create price series with some randomness
        np.random.seed(42)  # For reproducibility
        prices = np.cumsum(np.random.normal(0, 1, n)) * 0.5 + base_price
        volumes = np.random.randint(1000, 10000, n)

        # Create OHLCV data
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": prices * 1.01,  # 1% higher than close
                "low": prices * 0.99,  # 1% lower than close
                "close": prices,
                "volume": volumes,
            }
        )

        data[symbol] = df

    return data


@pytest.fixture
def context():
    """Create a strategy context."""
    return Context()


@pytest.fixture
def broker():
    """Create a broker instance."""
    return Broker(initial_cash=100000.0)


@pytest.fixture
def portfolio():
    """Create a portfolio instance."""
    return Portfolio(initial_cash=100000.0)


class SimpleTestStrategy(Strategy):
    """Simple strategy for testing."""

    def initialize(self, context):
        """Initialize test strategy."""
        context.symbols = ["AAPL", "MSFT", "GOOG"]

    def handle_data(self, context, data):
        """Generate simple signals."""
        signals = []

        for symbol in context.symbols:
            if symbol in data and len(data[symbol]) > 0:
                price = data[symbol]["close"].iloc[-1]

                # Buy if price is below 110, sell if above 130
                if price < 110:
                    signals.append(Signal(symbol=symbol, action="buy", quantity=10))
                elif price > 130:
                    signals.append(Signal(symbol=symbol, action="sell", quantity=-10))

        return signals


@pytest.fixture
def test_strategy():
    """Create a test strategy instance."""
    return SimpleTestStrategy()

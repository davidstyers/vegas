"""
Tests for Signal Research Extension

This module tests the signal research functionality including:
- Strategy predict() method
- BacktestEngine generate_signals() method  
- Alpha class for signal evaluation
"""

from datetime import datetime, timedelta
from typing import Dict

import polars as pl
import numpy as np
import pytest

from vegas.data.data_portal import DataPortal
from vegas.engine.engine import BacktestEngine
from vegas.strategy import Strategy
from vegas.analytics.alpha import Alpha


class MockDataLayer:
    """Mock data layer for testing."""
    
    def __init__(self, tz: str = "US/Eastern"):
        self.timezone = tz

    def get_data_for_backtest(self, start, end, symbols=None, market_hours=None):
        """Generate mock OHLCV data for testing."""
        base = datetime(2025, 1, 1, 9, 0, 0)
        ts = [base + timedelta(hours=i) for i in range(100)]  # 100 hours of data
        rows = []
        
        symbols_list = symbols if symbols else ["AAPL", "MSFT"]
        
        for sym in symbols_list:
            base_price = 100.0 if sym == "AAPL" else 200.0
            for i, t in enumerate(ts):
                # Create trending price movement for testing
                trend = 0.01 * i  # Small upward trend
                noise = 0.5 * np.sin(i * 0.1)  # Some cyclical noise
                price = base_price + trend + noise
                
                rows.append({
                    "timestamp": t,
                    "symbol": sym,
                    "open": price - 0.25,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000 + 10 * i,
                })
        return pl.from_dicts(rows)

    def get_unified_timestamp_index(self, start, end):
        """Return empty timestamp index for mock."""
        return pl.from_dicts({"timestamp": []}).get_column("timestamp")


class SimpleMovingAverageStrategy(Strategy):
    """Simple strategy that generates signals based on moving average crossover."""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.universe = ["AAPL", "MSFT"]
    
    def predict(self, context, data_portal) -> Dict[str, float]:
        """Generate signals based on moving average crossover."""
        signals = {}
        
        # Get universe from strategy or data portal
        symbols = getattr(self, 'universe', data_portal.get_symbols())
        
        for symbol in symbols:
            hist_data = data_portal.history(assets=[symbol], bar_count=self.long_window + 5)
            if not hist_data.is_empty() and hist_data.height >= self.long_window:
                # Calculate moving averages
                close_prices = hist_data.select("close").to_series().to_list()
                
                if len(close_prices) >= self.long_window:
                    short_ma = np.mean(close_prices[-self.short_window:])
                    long_ma = np.mean(close_prices[-self.long_window:])
                    
                    # Generate signal: positive when short MA > long MA
                    signal = (short_ma - long_ma) / long_ma  # Normalized signal
                    signals[symbol] = float(signal)
        
        return signals


class ConstantSignalStrategy(Strategy):
    """Strategy that returns constant signals for testing."""
    
    def __init__(self, signal_value: float = 0.5):
        super().__init__()
        self.signal_value = signal_value
        self.universe = ["AAPL"]
    
    def predict(self, context, data_portal) -> Dict[str, float]:
        """Return constant signal for available assets."""
        signals = {}
        symbols = getattr(self, 'universe', data_portal.get_symbols())
        for symbol in symbols:
            signals[symbol] = self.signal_value
        return signals


def test_strategy_predict_method():
    """Test that strategy predict method works correctly."""
    strategy = SimpleMovingAverageStrategy()
    
    # Create mock context and data portal
    from vegas.strategy import Context
    context = Context()
    context.current_ts = datetime(2025, 1, 1, 15, 0, 0)
    
    # Create mock data portal with history method
    class MockDataPortal:
        def get_symbols(self):
            return ["AAPL", "MSFT"]
        
        def history(self, assets, bar_count):
            # Return mock historical data
            timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(bar_count)]
            return pl.DataFrame({
                "timestamp": timestamps,
                "symbol": [assets[0]] * bar_count,
                "close": [100 + i * 0.1 for i in range(bar_count)]  # Trending up
            })
    
    data_portal = MockDataPortal()
    
    # Test predict method
    signals = strategy.predict(context, data_portal)
    
    assert isinstance(signals, dict)
    assert "AAPL" in signals or "MSFT" in signals
    if signals:
        assert all(isinstance(v, float) for v in signals.values())


def test_backtest_engine_generate_signals():
    """Test BacktestEngine generate_signals method."""
    # Create a simple test strategy that doesn't need historical data
    class SimpleTestStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.universe = ["TEST1", "TEST2"]
        
        def predict(self, context, data_portal) -> Dict[str, float]:
            # Simple strategy that returns fixed signals for universe
            return {"TEST1": 0.5, "TEST2": -0.3}
    
    # Setup engine (will use default data layer, but we'll mock the data portal)
    engine = BacktestEngine(data_dir="test_db")
    
    # Mock the data portal methods needed
    class MockDataPortal:
        def __init__(self):
            self.timezone = "UTC"
            
        def load_data(self, start_date, end_date, symbols=None, frequencies=None, calendar=None):
            pass
            
        def get_symbols(self):
            return ["TEST1", "TEST2"]
            
        def set_current_dt(self, dt):
            pass
            
        def get_unified_timestamp_index(self, start, end, frequency="1h"):
            # Return simple hourly timestamps
            timestamps = []
            current = start
            while current <= end:
                timestamps.append(current)
                current += timedelta(hours=1)
            return pl.Series("timestamp", timestamps)
    
    # Mock the data layer as well
    class MockDataLayer:
        def __init__(self):
            self.timezone = "UTC"
        
        def get_unified_timestamp_index(self, start, end):
            timestamps = []
            current = start
            while current <= end:
                timestamps.append(current)
                current += timedelta(hours=1)
            return pl.Series("timestamp", timestamps)
    
    # Replace engine's data portal and data layer
    engine.data_portal = MockDataPortal()
    engine.data_layer = MockDataLayer()
    
    strategy = SimpleTestStrategy()
    
    start = datetime(2025, 1, 1, 10)
    end = datetime(2025, 1, 1, 15)  # 6 hours of data
    
    # Generate signals
    signals_df = engine.generate_signals(start, end, strategy)
    
    # Validate output
    assert isinstance(signals_df, pl.DataFrame)
    assert "datetime" in signals_df.columns
    assert signals_df.height > 0
    
    # Should have columns for assets in the universe
    expected_symbols = set(strategy.universe)
    actual_symbols = set(signals_df.columns) - {"datetime"}
    assert expected_symbols.issubset(actual_symbols)


def test_alpha_forward_returns():
    """Test Alpha forward returns calculation."""
    # Create mock price data
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(20)]
    prices_df = pl.DataFrame({
        "datetime": timestamps,
        "AAPL": [100.0 + i for i in range(20)],  # Price going up by 1 each hour
        "MSFT": [200.0 + i * 0.5 for i in range(20)]  # Price going up by 0.5 each hour
    })
    
    # Create dummy signals (not used for forward returns calculation)
    signals_df = pl.DataFrame({
        "datetime": timestamps,
        "AAPL": [0.5] * 20,
        "MSFT": [0.3] * 20
    })
    
    # Create Alpha instance
    alpha = Alpha(signals_df, prices_df)
    
    # Calculate forward returns
    fwd_returns = alpha.forward_returns(horizons=[1, 5])
    
    # Validate results
    assert isinstance(fwd_returns, dict)
    assert 1 in fwd_returns
    assert 5 in fwd_returns
    
    # Check 1-period forward return
    fwd_1 = fwd_returns[1]
    assert isinstance(fwd_1, pl.DataFrame)
    assert "datetime" in fwd_1.columns
    assert "AAPL" in fwd_1.columns
    
    # For AAPL: price goes from 100 to 101, so 1-period return should be 0.01
    aapl_returns = fwd_1.filter(pl.col("datetime") == timestamps[0])["AAPL"].to_list()[0]
    expected_return = (101.0 - 100.0) / 100.0
    assert abs(aapl_returns - expected_return) < 1e-6


def test_alpha_evaluate():
    """Test Alpha evaluation with known signals and returns."""
    # Create test data where signals are positively correlated with future returns
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(50)]
    
    # Create prices with known pattern
    prices = []
    for i in range(50):
        base_price = 100.0
        # Add some trend and noise
        price = base_price + i * 0.5 + np.sin(i * 0.2) * 2
        prices.append(price)
    
    prices_df = pl.DataFrame({
        "datetime": timestamps,
        "AAPL": prices
    })
    
    # Create signals that should predict the price direction
    signals = []
    for i in range(50):
        if i < 49:
            # Signal based on next period price change
            signal = 1.0 if prices[i+1] > prices[i] else -1.0
        else:
            signal = 0.0
        signals.append(signal)
    
    signals_df = pl.DataFrame({
        "datetime": timestamps,
        "AAPL": signals
    })
    
    # Evaluate alpha
    alpha = Alpha(signals_df, prices_df)
    evaluation = alpha.evaluate(horizons=[1])
    
    # Validate results
    assert isinstance(evaluation, pl.DataFrame)
    assert evaluation.height == 1
    assert "horizon" in evaluation.columns
    assert "IC" in evaluation.columns
    assert "HitRate" in evaluation.columns
    
    # Hit rate should be very high since signals predict direction perfectly
    hit_rate = evaluation["HitRate"].to_list()[0]
    assert hit_rate > 0.9  # Should be close to 1.0


def test_single_asset_workflow():
    """Test complete single-asset workflow."""
    # Setup
    dl = MockDataLayer()
    engine = BacktestEngine(data_dir="test_db")
    engine.data_layer = dl
    engine.data_portal = DataPortal(dl)
    
    strategy = ConstantSignalStrategy(signal_value=0.8)
    
    start = datetime(2025, 1, 1, 10)
    end = datetime(2025, 1, 1, 20)
    
    # 1. Generate signals
    signals_df = engine.generate_signals(start, end, strategy)
    
    # 2. Get price data for alpha evaluation
    engine.data_portal.load_data(start, end, symbols=["AAPL"], frequencies=["1h"])
    price_data = engine.data_portal.history(
        assets=["AAPL"], 
        bar_count=1000, 
        frequency="1h", 
        end_dt=end
    )
    
    # Convert to datetime x assets format for Alpha
    prices_df = price_data.select(["timestamp", "close"]).rename({"timestamp": "datetime", "close": "AAPL"})
    
    # 3. Evaluate alpha
    alpha = Alpha(signals_df, prices_df)
    evaluation = alpha.evaluate(horizons=[1, 5])
    
    # Validate complete workflow
    assert isinstance(evaluation, pl.DataFrame)
    assert evaluation.height == 2  # Two horizons
    assert all(col in evaluation.columns for col in ["horizon", "IC", "HitRate"])


def test_multi_asset_dynamic_universe():
    """Test multi-asset strategy with dynamic universes."""
    
    class DynamicUniverseStrategy(Strategy):
        """Strategy with changing universe over time."""
        
        def __init__(self):
            super().__init__()
            self.universe = ["AAPL", "MSFT"]
        
        def predict(self, context, data_portal) -> Dict[str, float]:
            signals = {}
            
            # Use timestamp to determine which half we're in
            timestamp = context.current_ts
            # Simple logic: trade only AAPL before 2pm, both after
            if timestamp.hour < 14:
                signals["AAPL"] = 0.7
                # Note: Don't return signal for MSFT, should fill with None
            else:
                symbols = getattr(self, 'universe', ["AAPL", "MSFT"])
                for symbol in symbols:
                    signals[symbol] = 0.5
            
            return signals
    
    # Setup
    dl = MockDataLayer()
    engine = BacktestEngine(data_dir="test_db")
    engine.data_layer = dl
    engine.data_portal = DataPortal(dl)
    
    strategy = DynamicUniverseStrategy()
    
    start = datetime(2025, 1, 1, 10)
    end = datetime(2025, 1, 1, 18)  # 9 hours
    
    # Generate signals
    signals_df = engine.generate_signals(start, end, strategy)
    
    # Validate dynamic universe handling
    assert "AAPL" in signals_df.columns
    assert "MSFT" in signals_df.columns
    
    # Check that MSFT has None values in early periods
    early_msft_signals = signals_df.head(5)["MSFT"].to_list()
    none_count = sum(1 for x in early_msft_signals if x is None)
    assert none_count > 0  # Should have some None values for MSFT in early period


def test_alpha_consistency_toy_example():
    """Test alpha metrics consistency with a simple toy example."""
    # Create a more interesting price pattern with varying returns
    timestamps = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(10)]
    
    # Prices with varying returns: some up, some down
    # Pattern: up, up, down, up, down, down, up, up, down, up
    price_changes = [1.0, 2.0, -0.5, 1.5, -1.0, -0.8, 2.5, 1.2, -0.7, 1.8]
    prices = [100.0]
    for change in price_changes:
        prices.append(prices[-1] + change)
    
    prices_df = pl.DataFrame({
        "datetime": timestamps,
        "TEST": prices[:10]  # Take first 10 prices
    })
    
    # Create signals that correlate with future price movements
    # Signal should predict the direction of next period's price change
    future_changes = price_changes  # What actually happens next period
    signals = []
    for change in future_changes:
        if change > 0:
            signals.append(1.0)  # Bullish signal for up moves
        else:
            signals.append(-1.0)  # Bearish signal for down moves
    
    signals_df = pl.DataFrame({
        "datetime": timestamps,
        "TEST": signals
    })
    
    alpha = Alpha(signals_df, prices_df)
    evaluation = alpha.evaluate(horizons=[1])
    
    # With perfect directional correlation, hit rate should be 1.0
    ic = evaluation["IC"].to_list()[0]
    hit_rate = evaluation["HitRate"].to_list()[0]
    
    # IC should be positive (signals predict returns direction)
    assert ic > 0.5  # Strong positive correlation
    assert hit_rate == 1.0  # Perfect hit rate
    
    # Test zero correlation case - random signals
    random_signals = [0.1, -0.3, 0.7, -0.2, 0.8, -0.4, 0.2, 0.6, -0.1, 0.3]
    zero_signals_df = pl.DataFrame({
        "datetime": timestamps,
        "TEST": random_signals  # Random signals uncorrelated with returns
    })
    
    alpha_zero = Alpha(zero_signals_df, prices_df)
    evaluation_zero = alpha_zero.evaluate(horizons=[1])
    
    ic_zero = evaluation_zero["IC"].to_list()[0]
    hit_rate_zero = evaluation_zero["HitRate"].to_list()[0]
    
    # IC and hit rate should be somewhere in the middle for random signals
    assert abs(ic_zero) < 0.8  # Should not be perfectly correlated
    assert 0.2 <= hit_rate_zero <= 0.8  # Hit rate should be reasonable but not perfect


if __name__ == "__main__":
    pytest.main([__file__])

from datetime import datetime, timedelta

import polars as pl
import pytest

from vegas.analytics.alpha.engine import Alpha
from vegas.data.data_portal import DataPortal
from vegas.engine.engine import BacktestEngine
from vegas.strategy import Strategy


class MockDL:
    def __init__(self, tz="US/Eastern"):
        self.timezone = tz

    def get_data_for_backtest(self, start, end, symbols=None, market_hours=None):
        base = datetime(2025, 6, 2, 9, 0, 0)
        ts = [base + timedelta(hours=i) for i in range(8)]
        rows = []
        for sym in symbols or ["A", "B"]:
            for t in ts:
                rows.append(
                    {
                        "timestamp": t,
                        "symbol": sym,
                        "open": 1.0 + ts.index(t) * 0.1,
                        "high": 1.0 + ts.index(t) * 0.1,
                        "low": 1.0 + ts.index(t) * 0.1,
                        "close": 1.0 + ts.index(t) * 0.1,
                        "volume": 100,
                    }
                )
        df = pl.from_dicts(rows)
        return df.with_columns(
            pl.col("timestamp").cast(
                pl.Datetime(time_unit="us", time_zone=self.timezone)
            )
        )

    def get_unified_timestamp_index(self, start, end, frequency=None):
        base = datetime(2025, 6, 2, 9, 0, 0)
        ts = [base + timedelta(hours=i) for i in range(8)]
        return pl.Series("timestamp", ts, dtype=pl.Datetime(time_unit="us")).cast(
            pl.Datetime(time_unit="us", time_zone=self.timezone)
        )


class MockSignalStrategy(Strategy):
    """A strategy that generates predictable signals for testing."""

    def __init__(self, assets, signal_values):
        super().__init__()
        self.assets = assets
        self.signal_values = signal_values

    def predict(self, context, data):
        """Generates a mock signal DataFrame."""
        timestamps = data.select(pl.col("timestamp").unique()).sort("timestamp")
        signals = []
        for asset in self.assets:
            asset_signals = timestamps.with_columns(
                pl.lit(asset).alias("asset"),
                pl.Series("signal", self.signal_values),
            )
            signals.append(asset_signals)
        return pl.concat(signals)


def test_signal_generation_and_alpha_evaluation_integration():
    """
    Integration test for the full workflow from signal generation to alpha evaluation.
    """
    engine = BacktestEngine(timezone="US/Eastern")
    engine.data_layer = MockDL(tz="US/Eastern")
    engine.data_portal = DataPortal(engine.data_layer)

    assets = ["AAPL"]
    strategy = MockSignalStrategy(assets, [1.0] * 8)
    engine.strategy = strategy

    start_date = datetime(2025, 6, 2)
    end_date = datetime(2025, 6, 3)

    # 1. Generate signals
    engine.run(start_date, end_date, strategy)
    signals_df = engine.generate_signals()

    # 2. Evaluate signals with Alpha engine
    alpha = Alpha(
        signals_df,
        engine.data_layer.get_data_for_backtest(start_date, end_date, assets),
    )
    # The evaluate() method should run without errors
    alpha.evaluate()
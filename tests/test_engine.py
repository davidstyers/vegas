from datetime import datetime, timedelta

import polars as pl
import pytest

from vegas.data.data_portal import DataPortal
from vegas.engine.engine import BacktestEngine
from vegas.strategy import Strategy


class DummyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.calls = {
            "initialize": 0,
            "before_trading_start": 0,
            "handle_data": 0,
            "on_market_close": 0,
        }
        self.universe = ["A", "B"]

    def initialize(self, context):
        self.calls["initialize"] += 1

    def before_trading_start(self, context, data):
        self.calls["before_trading_start"] += 1

    def handle_data(self, context, data_portal):
        self.calls["handle_data"] += 1
        return []

    def on_market_close(self, context, data, portfolio):
        self.calls["on_market_close"] += 1


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
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
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

    def get_available_dates(self):
        return pl.DataFrame(
            {
                "day_count": [1],
                "start_date": [datetime(2025, 1, 1)],
                "end_date": [datetime(2025, 12, 31)],
            }
        )

    def load_data(self, *args, **kwargs):
        pass


def test_prepare_market_data_builds_index(monkeypatch):
    engine = BacktestEngine(timezone="US/Eastern")
    engine.strategy = DummyStrategy()
    mock_dl = MockDL(tz="US/Eastern")
    engine.data_layer = mock_dl
    engine.data_portal = DataPortal(mock_dl)

    from vegas.pipeline.pipeline import Pipeline

    p = Pipeline(columns={}, frequency="1h")
    engine.attach_pipeline(p, "noop")

    idx = engine._prepare_market_data(datetime(2025, 6, 2, 9), datetime(2025, 6, 2, 14))
    assert isinstance(idx, pl.Series)
    assert idx.dtype == pl.Datetime(time_unit="us", time_zone=engine.timezone)


def test_engine_initialization():
    engine = BacktestEngine(timezone="UTC")
    assert engine.timezone == "UTC"
    assert engine.portfolio is None


def test_engine_calendar_selection_from_cli_default():
    engine = BacktestEngine(timezone="US/Eastern")
    # Default calendar is 24/7 unless overridden by CLI; engine exposes _calendar_name
    assert getattr(engine, "_calendar_name", "24/7") == "24/7"


def test_calendar_filters_applied_via_portal(monkeypatch):
    from vegas.calendars import get_calendar

    engine = BacktestEngine(timezone="US/Eastern")
    engine._calendar_name = "NYSE"

    # Wire mocks
    engine.strategy = DummyStrategy()
    mock_dl = MockDL(tz="US/Eastern")
    engine.data_layer = mock_dl
    engine.data_portal = DataPortal(mock_dl)

    # Load with calendar and check timestamp index is non-empty and sorted
    idx = engine._prepare_market_data(datetime(2025, 6, 2, 8), datetime(2025, 6, 2, 18))
    assert isinstance(idx, pl.Series)
    assert idx.len() > 0
    assert idx.sort().to_list() == idx.to_list()


def test_attach_pipeline():
    from vegas.pipeline.pipeline import Pipeline

    engine = BacktestEngine(timezone="UTC")
    pipeline = Pipeline(columns={}, frequency="1h")
    engine.attach_pipeline(pipeline, "test_pipeline")
    assert "test_pipeline" in engine.attached_pipelines
    assert engine.attached_pipelines["test_pipeline"] is pipeline


def test_run_backtest_integration(monkeypatch):
    engine = BacktestEngine(timezone="US/Eastern")
    engine._calendar_name = "24/7"

    strategy = DummyStrategy()

    mock_dl = MockDL(tz="US/Eastern")
    engine.data_layer = mock_dl
    engine.data_portal = DataPortal(mock_dl)

    results = engine.run(
        start=datetime(2025, 6, 2),
        end=datetime(2025, 6, 3),
        strategy=strategy,
        initial_capital=10000.0,
    )

    assert results is not None
    assert strategy.calls["initialize"] == 1
    assert strategy.calls["before_trading_start"] > 0
    assert strategy.calls["handle_data"] > 0
    assert strategy.calls["on_market_close"] > 0
    assert results["stats"]["final_value"] == 10000.0

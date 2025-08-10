import polars as pl
from datetime import datetime, timedelta

from vegas.data.data_portal import DataPortal


class MockDataLayer:
    def __init__(self, tz: str = "US/Eastern"):
        self.timezone = tz

    def get_data_for_backtest(self, start, end, symbols=None, market_hours=None):
        # Build simple hourly bars for two symbols across 4 hours
        base = datetime(2025, 6, 2, 4, 0, 0)
        ts = [base + timedelta(hours=i) for i in range(4)]
        rows = []
        for sym, starts in [("A", 50.0), ("B", 100.0)]:
            for i, t in enumerate(ts):
                rows.append({
                    "timestamp": t,
                    "symbol": sym,
                    "open": starts + i,
                    "high": starts + i + 0.5,
                    "low": starts + i - 0.5,
                    "close": starts + i + 0.25,
                    "volume": 1000 + 10 * i,
                })
        return pl.from_dicts(rows)

    def get_unified_timestamp_index(self, start, end):
        return pl.from_dicts({"timestamp": []}).get_column("timestamp")


def test_load_and_slice_history_and_spot_value():
    dl = MockDataLayer()
    dp = DataPortal(dl)

    start = datetime(2025, 6, 2, 4)
    end = datetime(2025, 6, 2, 7)
    dp.load_data(start, end, symbols=["A", "B"], frequencies=["1h"]) 

    # get_slice_for_timestamp should return rows for both symbols
    ts = datetime(2025, 6, 2, 6)
    snap = dp.get_slice_for_timestamp(ts, symbols=None)
    assert not snap.is_empty()
    assert set(snap.get_column("symbol").to_list()) == {"A", "B"}

    # history rolling window per symbol with explicit end_dt
    dp.set_current_dt(end)
    h = dp.history(assets=["A", "B"], fields=["close"], bar_count=2, frequency="1h", end_dt=end)
    assert not h.is_empty()
    # Should have last two bars per symbol -> 4 rows total
    assert h.height == 4

    # spot value at a timestamp
    v = dp.get_spot_value("A", "close", ts, frequency="1h")
    assert isinstance(v, float)
    assert v > 0



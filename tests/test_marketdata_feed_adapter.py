import pytest
from datetime import datetime, timedelta, timezone
import polars as pl

from vegas.marketdata.feed import HistoricalFeedAdapter
from vegas.engine import BacktestEngine


class MockDataLayerPolars:
    """
    Minimal in-test DataLayer that returns a Polars DataFrame matching engine schema.
    Columns: timestamp (tz-aware), symbol, open, high, low, close, volume
    """

    def __init__(self, data: pl.DataFrame, timezone_name: str = "UTC"):
        self._data = data
        self.timezone = timezone_name

    def get_data_for_backtest(self, start, end, market_hours=None, symbols=None):
        df = self._data
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        df = df.filter((pl.col("timestamp") >= pl.lit(start)) & (pl.col("timestamp") <= pl.lit(end)))
        # Engine expects polars DataFrame
        return df.sort("timestamp")


def _build_dataset():
    # Build small deterministic dataset across 2 days, 3 timestamps per day, 2 symbols
    tz = timezone.utc
    base = datetime(2022, 1, 1, 9, 30, tzinfo=tz)
    timestamps = [base + timedelta(minutes=30 * i) for i in range(3)]
    timestamps += [base + timedelta(days=1, minutes=30 * i) for i in range(3)]
    symbols = ["AAA", "BBB"]
    rows = []
    for ts in timestamps:
        for sym in symbols:
            price = 100.0 + len(rows) * 0.1
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.2,
                    "volume": 1000,
                }
            )
    df = pl.DataFrame(rows)
    # Ensure correct dtypes
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")))
    return df


def test_historical_feed_adapter_iteration_equivalence():
    data = _build_dataset()
    mdl = MockDataLayerPolars(data)

    # Arrange adapter with same symbols and date range
    symbols = ["AAA", "BBB"]
    start = data.select(pl.col("timestamp").min()).item()
    end = data.select(pl.col("timestamp").max()).item()

    feed = HistoricalFeedAdapter(
        data_layer=mdl,
        symbols=symbols,
        start=start,
        end=end,
        market_hours=None,
        timezone="UTC",
    )
    feed.subscribe(symbols)

    # Act: iterate feed
    feed.start()
    seen = 0
    unique_ts = []
    while True:
        nxt = feed.next_bar()
        if nxt is None:
            break
        ts, ts_data = nxt
        assert isinstance(ts_data, pl.DataFrame)
        # expected symbols present at this timestamp
        syms_at_ts = set(ts_data.select("symbol").to_series().to_list()) if "symbol" in ts_data.columns else set()
        assert set(symbols).issubset(syms_at_ts)
        seen += 1
        unique_ts.append(ts)

    # Assert: count equals grouping the same way engine does (per timestamp bars)
    engine_like_group_count = (
        data.sort("timestamp")
        .group_by("timestamp", maintain_order=True)
        .len()
        .height
    )
    assert seen == engine_like_group_count
    # Verify is_realtime is false (historical)
    assert feed.is_realtime() is False

    # Confirm the unique timestamps match the engine grouping order
    grouped_ts = [k[0] for k, _ in data.sort("timestamp").group_by("timestamp", maintain_order=True)]
    assert unique_ts == grouped_ts
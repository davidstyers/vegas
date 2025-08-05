import polars as pl
from datetime import datetime, timedelta, timezone

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy
from vegas.marketdata.feed import HistoricalFeedAdapter


class MinimalModeStrategy(Strategy):
    def initialize(self, context):
        # Record mode observed during initialize
        self.init_mode = getattr(context, "mode", "backtest")
        context.recorded_modes = []

    def handle_data(self, context, data):
        # Record mode seen during handle_data
        context.recorded_modes.append(getattr(context, "mode", "backtest"))
        return []


class MockDataLayerPolars:
    def __init__(self, data: pl.DataFrame, timezone_name: str = "UTC"):
        self._data = data
        self.timezone = timezone_name

    def get_data_for_backtest(self, start, end, market_hours=None, symbols=None):
        df = self._data
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        df = df.filter((pl.col("timestamp") >= pl.lit(start)) & (pl.col("timestamp") <= pl.lit(end)))
        return df.sort("timestamp")


def _dataset_one_day_two_bars():
    tz = timezone.utc
    base = datetime(2022, 1, 1, 9, 30, tzinfo=tz)
    timestamps = [base, base + timedelta(minutes=30)]
    rows = []
    for ts in timestamps:
        rows.append(
            {
                "timestamp": ts,
                "symbol": "AAA",
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.2,
                "volume": 1000,
            }
        )
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    )
    return df


def test_context_mode_backtest_and_live():
    data = _dataset_one_day_two_bars()
    mdl = MockDataLayerPolars(data)

    # Backtest path via run()
    engine_bt = BacktestEngine()
    engine_bt.data_layer = mdl
    strat_bt = MinimalModeStrategy()

    start = data.select(pl.col("timestamp").min()).item()
    end = data.select(pl.col("timestamp").max()).item()

    engine_bt.run(start, end, strat_bt, initial_capital=100000.0)
    # Expect default/backtest mode throughout when using run()
    assert strat_bt.init_mode != "live"
    assert all(mode != "live" for mode in strat_bt.context.recorded_modes)

    # Live path via run_live() with HistoricalFeedAdapter
    engine_live = BacktestEngine()
    engine_live.data_layer = mdl
    strat_live = MinimalModeStrategy()

    feed = HistoricalFeedAdapter(
        data_layer=mdl,
        symbols=["AAA"],
        start=start,
        end=end,
        market_hours=None,
        timezone="UTC",
    )

    engine_live.run_live(start, end, strat_live, feed=feed, broker=None)
    # Context.mode should be "live"
    assert getattr(strat_live.context, "mode", None) == "live"
    # Also recorded during handle_data
    assert len(strat_live.context.recorded_modes) > 0
    assert all(mode == "live" for mode in strat_live.context.recorded_modes)
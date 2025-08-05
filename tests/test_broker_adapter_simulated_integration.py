import pytest
from datetime import datetime, timedelta, timezone
import polars as pl

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Signal
from vegas.broker.broker import Broker
from vegas.broker.adapters import SimulatedBrokerAdapter
from vegas.marketdata.feed import HistoricalFeedAdapter


class MockDataLayerPolars:
    """
    Minimal in-test DataLayer returning Polars DataFrame.
    """

    def __init__(self, data: pl.DataFrame, timezone_name: str = "UTC"):
        self._data = data
        self.timezone = timezone_name

    def get_data_for_backtest(self, start, end, market_hours=None, symbols=None):
        df = self._data
        if symbols:
            df = df.filter(pl.col("symbol").is_in(symbols))
        df = df.filter((pl.col("timestamp") >= pl.lit(start)) & (pl.col("timestamp") <= pl.lit(end)))
        return df.sort("timestamp")


def _build_dataset_two_days_two_symbols():
    tz = timezone.utc
    base = datetime(2022, 1, 1, 9, 30, tzinfo=tz)
    timestamps = [base + timedelta(minutes=30 * i) for i in range(2)]
    symbols = ["AAA", "BBB"]
    rows = []
    price_seed = 100.0
    for ts in timestamps:
        for sym in symbols:
            price_seed += 0.1
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": price_seed,
                    "high": price_seed + 0.5,
                    "low": price_seed - 0.5,
                    "close": price_seed + 0.2,
                    "volume": 1000,
                }
            )
    df = pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    )
    return df


class SingleBuyStrategy(Strategy):
    """
    Places a single market buy for AAA on the first handle_data call, then nothing else.
    """

    def initialize(self, context):
        context.did_buy = False
        context.symbols = ["AAA", "BBB"]

    def handle_data(self, context, data):
        if not context.did_buy:
            # Ensure AAA is available in the slice
            if "symbol" in data.columns and "AAA" in set(data.select("symbol").to_series().to_list()):
                context.did_buy = True
                # Use market order (price=None) or set to data close for determinism
                price = None
                return [Signal(symbol="AAA", action="buy", quantity=10, price=price)]
        return []


def test_historical_iteration_with_simulated_broker_adapter_via_run_live_delegate():
    """
    Case A variant: Engine run_live with no feed should delegate to historical run(),
    ensuring default behavior remains unchanged, but we provide a SimulatedBrokerAdapter
    as 'broker' argument. This verifies adapter path does not break delegation.
    """
    # Arrange engine with deterministic dataset
    data = _build_dataset_two_days_two_symbols()
    engine = BacktestEngine()
    engine.data_layer = MockDataLayerPolars(data)

    strategy = SingleBuyStrategy()
    broker = Broker(initial_cash=100000.0)
    adapter = SimulatedBrokerAdapter(broker)

    start = data.select(pl.col("timestamp").min()).item()
    end = data.select(pl.col("timestamp").max()).item()

    # Act: run_live without feed - should delegate to run()
    results = engine.run_live(start, end, strategy, feed=None, broker=adapter)

    # Assert: portfolio reflects a single executed trade; transactions non-empty
    tx = results["transactions"]
    assert tx.height >= 1
    # Expect at least one AAA buy
    assert "symbol" in tx.columns
    assert any(row["symbol"] == "AAA" and row["quantity"] > 0 for row in tx.to_dicts())

    # Open orders in adapter should be empty or filled after execution
    open_orders = adapter.get_open_orders()
    assert isinstance(open_orders, list)
    # No requirement they must be empty in historical path, but typically filled
    # We assert they are not in a bad type
    # Portfolio cash decreased
    stats = results["stats"]
    assert stats["cash"] < 100000.0 or stats["num_trades"] >= 1


def test_live_iteration_with_historical_feed_and_simulated_broker_adapter():
    """
    Case B: run_live with HistoricalFeedAdapter and SimulatedBrokerAdapter should:
      - set Context.mode == "live"
      - route orders via adapter.place_order + simulate_execute per bar
      - update portfolio accordingly and stop after feed exhaustion
    """
    # Arrange
    data = _build_dataset_two_days_two_symbols()
    mdl = MockDataLayerPolars(data)
    engine = BacktestEngine()
    engine.data_layer = mdl

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

    broker = Broker(initial_cash=100000.0)
    adapter = SimulatedBrokerAdapter(broker)
    strategy = SingleBuyStrategy()

    # Act
    results = engine.run_live(start, end, strategy, feed=feed, broker=adapter)

    # Assert context.mode propagated
    assert getattr(strategy.context, "mode", None) == "live"

    # Verify at least one transaction produced via simulated path
    tx = results["transactions"]
    assert tx.height >= 1
    assert any(row["symbol"] == "AAA" and row["quantity"] > 0 for row in tx.to_dicts())

    # Portfolio updated; there should be a position or at least cash moved
    positions_df = engine.portfolio.get_positions_dataframe()
    # Either we hold AAA or have recorded buy then mark-to-market updates
    assert ("AAA" in set(positions_df["symbol"].to_list())) or (results["stats"]["num_trades"] >= 1)

    # Ensure loop terminated at feed exhaustion by comparing equity history timestamps to input range
    eq = results["equity_curve"]
    assert eq.height > 0
    last_eq_ts = eq.select(pl.col("timestamp").max()).item()
    assert last_eq_ts <= end
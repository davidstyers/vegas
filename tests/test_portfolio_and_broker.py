import polars as pl
from datetime import datetime

from vegas.portfolio.portfolio import Portfolio
from vegas.broker.broker import Broker, OrderStatus


class MockDataPortal:
    def __init__(self):
        self.timezone = 'US/Eastern'

    def get_slice_for_timestamp(self, timestamp, symbols=None, market_hours=None):
        # Return a simple 1-row per symbol slice with close prices
        rows = []
        for s in (symbols or []):
            rows.append({"timestamp": timestamp, "symbol": s, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100})
        return pl.from_dicts(rows)

    def get_spot_value(self, asset, field, dt, frequency='1h'):
        return 10.0


def test_broker_execute_orders_and_portfolio_update():
    dp = MockDataPortal()
    broker = Broker(initial_cash=1000.0, data_portal=dp)
    portfolio = Portfolio(initial_capital=1000.0, data_portal=dp)

    # Place a market order and execute
    from vegas.strategy import Signal
    signal = Signal(symbol='AAPL', quantity=10, order_type='market')
    order = broker.place_order(signal)
    assert order.status == OrderStatus.OPEN

    ts = datetime(2025, 6, 2, 10)
    txns = broker.execute_orders_with_portal(['AAPL'], ts)
    assert len(txns) == 1

    # Update portfolio from txns
    broker.update_market_values({})
    portfolio.update_from_transactions(ts, pl.from_dicts([{ 'symbol': 'AAPL', 'quantity': 10, 'price': 10.0, 'commission': 0.0 }]))
    assert 'AAPL' in portfolio.positions
    assert portfolio.get_portfolio_value() <= 1000.0  # spent cash



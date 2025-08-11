from datetime import datetime

import polars as pl

from vegas.broker.broker import Broker, OrderStatus


class MockDataPortal:
    def __init__(self):
        self.timezone = "US/Eastern"

    def get_slice_for_timestamp(self, timestamp, symbols=None, market_hours=None):
        # Return a simple 1-row per symbol slice with close prices
        rows = []
        for s in symbols or []:
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": s,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100,
                }
            )
        return pl.from_dicts(rows)

    def get_spot_value(self, asset, field, dt, frequency="1h"):
        return 10.0


def test_broker_execute_orders():
    dp = MockDataPortal()
    broker = Broker(initial_cash=1000.0, data_portal=dp)

    # Place a market order and execute
    from vegas.strategy import Signal

    signal = Signal(symbol="AAPL", quantity=10, order_type="market")
    order = broker.place_order(signal)
    assert order.status == OrderStatus.OPEN

    ts = datetime(2025, 6, 2, 10)
    txns = broker.execute_orders_with_portal(["AAPL"], ts)
    assert len(txns) == 1


from vegas.strategy import Signal


def test_order_creation():
    # Arrange
    broker = Broker(data_portal=MockDataPortal())
    signal = Signal(symbol="AAPL", quantity=10, order_type="market")
    # Act
    order = broker.place_order(signal)
    # Assert
    assert order.symbol == "AAPL"
    assert order.quantity == 10
    assert order.status == OrderStatus.OPEN
    assert len(broker.orders) == 1


def test_broker_initialization():
    # Arrange & Act
    broker = Broker(initial_cash=5000.0, data_portal=MockDataPortal())
    # Assert
    assert broker.cash == 5000.0
    assert len(broker.orders) == 0


def test_cancel_order():
    # Arrange
    dp = MockDataPortal()
    broker = Broker(initial_cash=1000.0, data_portal=dp)
    signal = Signal(symbol="AAPL", quantity=10, order_type="market")
    order = broker.place_order(signal)
    # Act
    broker.cancel_order(order.id)
    # Assert
    assert order.status == OrderStatus.CANCELLED
    open_orders = [o for o in broker.orders if o.status == OrderStatus.OPEN]
    assert len(open_orders) == 0


def test_order_lifecycle_integration():
    # Arrange
    dp = MockDataPortal()
    broker = Broker(initial_cash=1000.0, data_portal=dp)
    signal = Signal(symbol="MSFT", quantity=5, order_type="market")

    # Act: Place Order
    order = broker.place_order(signal)

    # Assert: Order is Open
    open_orders = [o for o in broker.orders if o.status == OrderStatus.OPEN]
    assert order.id in [o.id for o in open_orders]
    assert order.status == OrderStatus.OPEN

    # Act: Execute Order
    ts = datetime(2025, 6, 3, 11)
    transactions = broker.execute_orders_with_portal(["MSFT"], ts)

    # Assert: Order is Filled and Transaction is Created
    open_orders = [o for o in broker.orders if o.status == OrderStatus.OPEN]
    assert len(open_orders) == 0
    assert order.status == OrderStatus.FILLED
    assert len(transactions) == 1
    assert transactions[0].symbol == "MSFT"
    assert transactions[0].quantity == 5

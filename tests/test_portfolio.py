from datetime import datetime

import polars as pl

from vegas.portfolio.portfolio import Portfolio


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


def test_portfolio_update():
    dp = MockDataPortal()
    portfolio = Portfolio(initial_capital=1000.0, data_portal=dp)
    ts = datetime(2025, 6, 2, 10)
    # Update portfolio from txns
    portfolio.update_from_transactions(
        ts,
        pl.from_dicts(
            [{"symbol": "AAPL", "quantity": 10, "price": 10.0, "commission": 0.0}]
        ),
    )
    assert "AAPL" in portfolio.positions
    assert portfolio.get_portfolio_value() <= 1000.0  # spent cash


def test_portfolio_initialization():
    # Arrange & Act
    portfolio = Portfolio(initial_capital=25000.0, data_portal=MockDataPortal())
    # Assert
    assert portfolio.initial_capital == 25000.0
    assert portfolio.get_portfolio_value() == 25000.0
    assert len(portfolio.positions) == 0


def test_update_from_transactions_buy():
    # Arrange
    dp = MockDataPortal()
    portfolio = Portfolio(initial_capital=1000.0, data_portal=dp)
    ts = datetime(2025, 6, 2, 10)
    transactions = pl.from_dicts(
        [{"symbol": "GOOG", "quantity": 5, "price": 150.0, "commission": 1.0}]
    )

    # Act
    portfolio.update_from_transactions(ts, transactions)

    # Assert
    assert "GOOG" in portfolio.positions
    assert portfolio.positions["GOOG"] == 5
    assert portfolio.avg_price["GOOG"] == 150.0
    assert portfolio.current_cash == 1000.0 - (5 * 150.0) - 1.0


def test_portfolio_state_changes_over_time_integration():
    # Arrange
    dp = MockDataPortal()
    portfolio = Portfolio(initial_capital=10000.0, data_portal=dp)

    # Day 1: Buy AAPL
    ts1 = datetime(2025, 6, 2, 10)
    txns1 = pl.from_dicts(
        [{"symbol": "AAPL", "quantity": 10, "price": 100.0, "commission": 1.0}]
    )
    portfolio.update_from_transactions(ts1, txns1)

    # Assert Day 1
    assert portfolio.positions["AAPL"] == 10
    assert portfolio.current_cash == 10000.0 - 1001.0

    # Day 2: Sell AAPL
    ts2 = datetime(2025, 6, 3, 14)
    txns2 = pl.from_dicts(
        [{"symbol": "AAPL", "quantity": -5, "price": 110.0, "commission": 1.0}]
    )
    portfolio.update_from_transactions(ts2, txns2)

    # Assert Day 2
    assert portfolio.positions["AAPL"] == 5
    assert portfolio.current_cash == 10000.0 - 1001.0 + (5 * 110.0) - 1.0
    assert portfolio.get_portfolio_value() > 9000.0  # Should be profitable

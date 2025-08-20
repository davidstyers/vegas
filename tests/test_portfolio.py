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


def test_get_stats_with_quantstats():
    """Test that get_stats method works with quantstats integration."""
    # Arrange
    dp = MockDataPortal()
    portfolio = Portfolio(initial_capital=10000.0, data_portal=dp)

    # Create some trading activity to generate returns
    ts1 = datetime(2025, 6, 2, 10)
    txns1 = pl.from_dicts(
        [{"symbol": "AAPL", "quantity": 10, "price": 100.0, "commission": 1.0}]
    )
    portfolio.update_from_transactions(ts1, txns1)

    ts2 = datetime(2025, 6, 3, 14)
    txns2 = pl.from_dicts(
        [{"symbol": "AAPL", "quantity": -5, "price": 110.0, "commission": 1.0}]
    )
    portfolio.update_from_transactions(ts2, txns2)

    # Act
    stats = portfolio.get_stats()

    # Assert - check that core stats are present
    assert "initial_capital" in stats
    assert "final_value" in stats
    assert "total_return" in stats
    assert "total_return_pct" in stats
    assert "num_trades" in stats
    assert "sharpe_ratio" in stats
    assert "annual_return_pct" in stats
    assert "max_drawdown_pct" in stats
    
    # Check that new quantstats-based metrics are present
    assert "volatility_pct" in stats
    assert "sortino_ratio" in stats
    assert "calmar_ratio" in stats
    assert "var_95" in stats
    assert "cvar_95" in stats
    assert "win_rate" in stats
    assert "profit_factor" in stats
    assert "expectancy" in stats
    assert "best_day" in stats
    assert "worst_day" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats

    # Verify basic calculations
    assert stats["initial_capital"] == 10000.0
    assert stats["num_trades"] == 2
    assert isinstance(stats["sharpe_ratio"], (int, float))
    assert isinstance(stats["max_drawdown_pct"], (int, float))


def test_get_stats_with_insufficient_data():
    """Test that get_stats method handles insufficient data gracefully."""
    # Arrange - portfolio with no trading activity
    dp = MockDataPortal()
    portfolio = Portfolio(initial_capital=10000.0, data_portal=dp)

    # Act
    stats = portfolio.get_stats()

    # Assert - check that core stats are present with default values
    assert "initial_capital" in stats
    assert "final_value" in stats
    assert "total_return" in stats
    assert "total_return_pct" in stats
    assert "num_trades" in stats
    assert "sharpe_ratio" in stats
    assert "annual_return_pct" in stats
    assert "max_drawdown_pct" in stats
    
    # Check that new quantstats-based metrics are present with default values
    assert "volatility_pct" in stats
    assert "sortino_ratio" in stats
    assert "calmar_ratio" in stats
    assert "var_95" in stats
    assert "cvar_95" in stats
    assert "win_rate" in stats
    assert "profit_factor" in stats
    assert "expectancy" in stats
    assert "best_day" in stats
    assert "worst_day" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats

    # Verify default values for insufficient data
    assert stats["initial_capital"] == 10000.0
    assert stats["final_value"] == 10000.0
    assert stats["total_return"] == 0.0
    assert stats["total_return_pct"] == 0.0
    assert stats["num_trades"] == 0
    assert stats["sharpe_ratio"] == 0.0
    assert stats["annual_return_pct"] == 0.0
    assert stats["max_drawdown_pct"] == 0.0
    assert stats["volatility_pct"] == 0.0
    assert stats["sortino_ratio"] == 0.0
    assert stats["calmar_ratio"] == 0.0
    assert stats["var_95"] == 0.0
    assert stats["cvar_95"] == 0.0
    assert stats["win_rate"] == 0.0
    assert stats["profit_factor"] == 0.0
    assert stats["expectancy"] == 0.0
    assert stats["best_day"] == 0.0
    assert stats["worst_day"] == 0.0
    assert stats["skewness"] == 0.0
    assert stats["kurtosis"] == 0.0

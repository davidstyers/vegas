"""Test for long-only strategy validation.

This tests that the portfolio properly constrains transactions for a long-only strategy.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from vegas.portfolio.portfolio import Portfolio


def test_long_only_transaction_validation():
    """Test that the portfolio prevents short selling and buying beyond cash balance."""
    # Create a portfolio with $10,000 initial cash
    portfolio = Portfolio(initial_capital=10000.0)
    
    # Create some market data
    market_data = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'close': 150.0,
            'open': 149.0,
            'high': 151.0,
            'low': 148.0,
            'volume': 1000000
        },
        {
            'symbol': 'GOOG',
            'close': 2500.0,
            'open': 2480.0,
            'high': 2520.0,
            'low': 2470.0,
            'volume': 500000
        }
    ])
    
    # Test 1: Buy within cash limits
    timestamp = datetime(2022, 1, 1)
    buy_transactions = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': 50,  # 50 * $150 = $7,500 (within $10,000 cash)
            'price': 150.0,
            'commission': 7.5
        }
    ])
    
    portfolio.update_from_transactions(timestamp, buy_transactions, market_data)
    
    # Should have bought all 50 shares
    assert portfolio.positions['AAPL'] == 50
    # Cash should be reduced by $7,507.5 (cost + commission)
    assert pytest.approx(portfolio.current_cash, 0.01) == 10000 - (50 * 150 + 7.5)
    
    # Test 2: Try to buy more than we can afford
    timestamp = datetime(2022, 1, 2)
    expensive_buy = pd.DataFrame([
        {
            'symbol': 'GOOG',
            'quantity': 10,  # 10 * $2,500 = $25,000 (beyond remaining cash)
            'price': 2500.0,
            'commission': 25.0
        }
    ])
    
    portfolio.update_from_transactions(timestamp, expensive_buy, market_data)
    
    # Should have bought only what we could afford (about 1 share)
    assert 'GOOG' in portfolio.positions
    assert portfolio.positions['GOOG'] < 10  # Less than requested
    assert portfolio.positions['GOOG'] > 0  # But should have bought some
    
    # Cash should be almost depleted but not negative
    assert portfolio.current_cash >= 0
    assert portfolio.current_cash < 100  # Should be close to zero
    
    # Test 3: Try to sell more than we own
    timestamp = datetime(2022, 1, 3)
    oversell = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': -100,  # We only have 50 shares
            'price': 150.0,
            'commission': 7.5
        }
    ])
    
    initial_cash = portfolio.current_cash
    portfolio.update_from_transactions(timestamp, oversell, market_data)
    
    # Should have sold only what we had (50 shares)
    assert 'AAPL' not in portfolio.positions  # All shares sold
    
    # Cash should increase by 50 * $150 - commission
    expected_increase = 50 * 150 - 7.5
    assert pytest.approx(portfolio.current_cash - initial_cash, 0.01) == expected_increase
    
    # Test 4: Try to sell a stock we don't own
    timestamp = datetime(2022, 1, 4)
    invalid_sell = pd.DataFrame([
        {
            'symbol': 'MSFT',  # We don't own any MSFT
            'quantity': -10,
            'price': 300.0,
            'commission': 3.0
        }
    ])
    
    initial_cash = portfolio.current_cash
    portfolio.update_from_transactions(timestamp, invalid_sell, market_data)
    
    # Should not have sold anything
    assert 'MSFT' not in portfolio.positions
    
    # Cash should remain unchanged
    assert portfolio.current_cash == initial_cash 
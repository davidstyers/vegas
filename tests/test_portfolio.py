"""Tests for the portfolio functionality of the Vegas backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from vegas.portfolio import Portfolio


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with initial capital."""
    return Portfolio(initial_capital=100000.0)


@pytest.fixture
def sample_transactions():
    """Create sample transactions data."""
    return pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'commission': 7.5
        },
        {
            'symbol': 'MSFT',
            'quantity': 50,
            'price': 250.0,
            'commission': 7.5
        },
        {
            'symbol': 'GOOG',
            'quantity': 20,
            'price': 2500.0,
            'commission': 7.5
        }
    ])


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return pd.DataFrame([
        {
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        },
        {
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'symbol': 'MSFT',
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        },
        {
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'symbol': 'GOOG',
            'open': 2500.0,
            'high': 2520.0,
            'low': 2490.0,
            'close': 2510.0,
            'volume': 100000
        }
    ])


def test_portfolio_initialization(sample_portfolio):
    """Test portfolio initialization."""
    assert sample_portfolio.initial_capital == 100000.0
    assert sample_portfolio.current_cash == 100000.0
    assert sample_portfolio.current_equity == 100000.0
    assert sample_portfolio.positions == {}
    assert sample_portfolio.position_values == {}
    assert len(sample_portfolio.equity_history) == 0
    assert len(sample_portfolio.position_history) == 0
    assert len(sample_portfolio.transaction_history) == 0


def test_position_tracking_and_updates(sample_portfolio, sample_transactions, sample_market_data):
    """Test position tracking and updates."""
    # Initial state
    assert len(sample_portfolio.positions) == 0
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Check positions
    assert len(sample_portfolio.positions) == 3
    assert 'AAPL' in sample_portfolio.positions
    assert 'MSFT' in sample_portfolio.positions
    assert 'GOOG' in sample_portfolio.positions
    
    assert sample_portfolio.positions['AAPL'] == 100
    assert sample_portfolio.positions['MSFT'] == 50
    assert sample_portfolio.positions['GOOG'] == 20
    
    # Check position values
    assert 'AAPL' in sample_portfolio.position_values
    assert 'MSFT' in sample_portfolio.position_values
    assert 'GOOG' in sample_portfolio.position_values
    
    assert sample_portfolio.position_values['AAPL'] == 100 * 151.0
    assert sample_portfolio.position_values['MSFT'] == 50 * 251.0
    assert sample_portfolio.position_values['GOOG'] == 20 * 2510.0
    
    # Test position history
    assert len(sample_portfolio.position_history) == 3
    
    # Test selling positions
    sell_transactions = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': -50,  # Sell half
            'price': 152.0,
            'commission': 7.5
        }
    ])
    
    new_timestamp = timestamp + timedelta(hours=1)
    new_market_data = pd.DataFrame([
        {
            'timestamp': new_timestamp,
            'symbol': 'AAPL',
            'open': 151.0,
            'high': 153.0,
            'low': 150.0,
            'close': 152.0,
            'volume': 1000000
        },
        {
            'timestamp': new_timestamp,
            'symbol': 'MSFT',
            'open': 251.0,
            'high': 253.0,
            'low': 250.0,
            'close': 252.0,
            'volume': 500000
        },
        {
            'timestamp': new_timestamp,
            'symbol': 'GOOG',
            'open': 2510.0,
            'high': 2530.0,
            'low': 2500.0,
            'close': 2520.0,
            'volume': 100000
        }
    ])
    
    sample_portfolio.update_from_transactions(new_timestamp, sell_transactions, new_market_data)
    
    # Check updated positions
    assert sample_portfolio.positions['AAPL'] == 50  # Reduced by half
    assert sample_portfolio.positions['MSFT'] == 50  # Unchanged
    assert sample_portfolio.positions['GOOG'] == 20  # Unchanged
    
    # Test closing a position completely
    close_transactions = pd.DataFrame([
        {
            'symbol': 'MSFT',
            'quantity': -50,  # Sell all
            'price': 253.0,
            'commission': 7.5
        }
    ])
    
    final_timestamp = new_timestamp + timedelta(hours=1)
    final_market_data = pd.DataFrame([
        {
            'timestamp': final_timestamp,
            'symbol': 'AAPL',
            'open': 152.0,
            'high': 154.0,
            'low': 151.0,
            'close': 153.0,
            'volume': 1000000
        },
        {
            'timestamp': final_timestamp,
            'symbol': 'MSFT',
            'open': 252.0,
            'high': 254.0,
            'low': 251.0,
            'close': 253.0,
            'volume': 500000
        },
        {
            'timestamp': final_timestamp,
            'symbol': 'GOOG',
            'open': 2520.0,
            'high': 2540.0,
            'low': 2510.0,
            'close': 2530.0,
            'volume': 100000
        }
    ])
    
    sample_portfolio.update_from_transactions(final_timestamp, close_transactions, final_market_data)
    
    # Check updated positions
    assert 'MSFT' not in sample_portfolio.positions  # Position closed
    assert sample_portfolio.positions['AAPL'] == 50  # Unchanged
    assert sample_portfolio.positions['GOOG'] == 20  # Unchanged


def test_cash_balance_management(sample_portfolio, sample_transactions, sample_market_data):
    """Test cash balance management."""
    # Initial cash
    initial_cash = sample_portfolio.current_cash
    
    # Calculate expected cash after transactions
    expected_cash_used = (
        100 * 150.0 +  # AAPL
        50 * 250.0 +   # MSFT
        20 * 2500.0 +  # GOOG
        7.5 * 3        # Commissions
    )
    expected_cash = initial_cash - expected_cash_used
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Check cash balance
    assert sample_portfolio.current_cash == pytest.approx(expected_cash)
    
    # Test cash after selling
    sell_transactions = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': -50,  # Sell half
            'price': 152.0,
            'commission': 7.5
        }
    ])
    
    expected_cash_added = 50 * 152.0 - 7.5  # Sale proceeds minus commission
    expected_cash = sample_portfolio.current_cash + expected_cash_added
    
    new_timestamp = timestamp + timedelta(hours=1)
    new_market_data = sample_market_data.copy()
    new_market_data['timestamp'] = new_timestamp
    new_market_data.loc[new_market_data['symbol'] == 'AAPL', 'close'] = 152.0
    
    sample_portfolio.update_from_transactions(new_timestamp, sell_transactions, new_market_data)
    
    # Check updated cash balance
    assert sample_portfolio.current_cash == pytest.approx(expected_cash)


def test_equity_curve_calculation(sample_portfolio):
    """Test equity curve calculation."""
    # Initial state
    assert len(sample_portfolio.equity_history) == 0
    
    # Create a series of transactions and market data updates
    timestamps = [
        datetime(2022, 1, 1, 9, 30),
        datetime(2022, 1, 1, 10, 30),
        datetime(2022, 1, 1, 11, 30),
        datetime(2022, 1, 1, 12, 30)
    ]
    
    # Buy transactions
    buy_transactions = pd.DataFrame([
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'commission': 7.5
        }
    ])
    
    # Market data with price changes
    market_data_series = [
        pd.DataFrame([{
            'timestamp': timestamps[0],
            'symbol': 'AAPL',
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }]),
        pd.DataFrame([{
            'timestamp': timestamps[1],
            'symbol': 'AAPL',
            'open': 151.0,
            'high': 153.0,
            'low': 150.0,
            'close': 152.0,
            'volume': 1000000
        }]),
        pd.DataFrame([{
            'timestamp': timestamps[2],
            'symbol': 'AAPL',
            'open': 152.0,
            'high': 154.0,
            'low': 151.0,
            'close': 153.0,
            'volume': 1000000
        }]),
        pd.DataFrame([{
            'timestamp': timestamps[3],
            'symbol': 'AAPL',
            'open': 153.0,
            'high': 155.0,
            'low': 152.0,
            'close': 154.0,
            'volume': 1000000
        }])
    ]
    
    # Execute initial transaction
    sample_portfolio.update_from_transactions(timestamps[0], buy_transactions, market_data_series[0])
    
    # Update with market data only for subsequent timestamps
    for i in range(1, len(timestamps)):
        sample_portfolio.update_from_transactions(timestamps[i], pd.DataFrame(), market_data_series[i])
    
    # Get equity curve
    equity_curve = sample_portfolio.get_equity_curve()
    
    # Verify equity curve
    assert not equity_curve.empty
    assert len(equity_curve) == len(timestamps)
    assert list(equity_curve['timestamp']) == timestamps
    
    # Verify equity values increase as stock price increases
    equity_values = equity_curve['equity'].tolist()
    assert all(equity_values[i] <= equity_values[i+1] for i in range(len(equity_values)-1))
    
    # Verify returns calculation
    returns = sample_portfolio.get_returns()
    assert not returns.empty
    assert len(returns) == len(timestamps)
    assert 'return' in returns.columns
    
    # First return should be 0 (or NaN)
    assert returns['return'].iloc[0] == 0
    
    # Subsequent returns should be positive as stock price increases
    assert all(returns['return'].iloc[i] >= 0 for i in range(1, len(returns)))


def test_transaction_recording(sample_portfolio, sample_transactions, sample_market_data):
    """Test transaction recording."""
    # Initial state
    assert len(sample_portfolio.transaction_history) == 0
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Check transaction history
    assert len(sample_portfolio.transaction_history) == len(sample_transactions)
    
    # Verify transaction details
    for i, expected_txn in enumerate(sample_transactions.to_dict('records')):
        actual_txn = sample_portfolio.transaction_history[i]
        assert actual_txn['symbol'] == expected_txn['symbol']
        assert actual_txn['quantity'] == expected_txn['quantity']
        assert actual_txn['price'] == expected_txn['price']
        assert actual_txn['commission'] == expected_txn['commission']
        assert actual_txn['timestamp'] == timestamp
    
    # Get transactions as DataFrame
    txn_df = sample_portfolio.get_transactions()
    assert not txn_df.empty
    assert len(txn_df) == len(sample_transactions)
    assert set(txn_df.columns) >= {'timestamp', 'symbol', 'quantity', 'price', 'commission'}


def test_performance_statistics_calculation(sample_portfolio, sample_transactions, sample_market_data):
    """Test performance statistics calculation."""
    # Initial state
    initial_stats = sample_portfolio.get_stats()
    assert initial_stats['total_return'] == 0.0
    assert initial_stats['total_return_pct'] == 0.0
    assert initial_stats['num_trades'] == 0
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Check stats after transactions
    stats_after_txn = sample_portfolio.get_stats()
    assert stats_after_txn['num_trades'] == len(sample_transactions)
    
    # Calculate expected return
    initial_capital = sample_portfolio.initial_capital
    current_equity = sample_portfolio.current_equity
    expected_return = current_equity - initial_capital
    expected_return_pct = (expected_return / initial_capital) * 100.0
    
    assert stats_after_txn['total_return'] == pytest.approx(expected_return)
    assert stats_after_txn['total_return_pct'] == pytest.approx(expected_return_pct)
    
    # Update with market data changes
    new_timestamp = timestamp + timedelta(hours=1)
    new_market_data = sample_market_data.copy()
    new_market_data['timestamp'] = new_timestamp
    
    # Increase all prices by 5%
    for symbol in ['AAPL', 'MSFT', 'GOOG']:
        new_market_data.loc[new_market_data['symbol'] == symbol, 'close'] *= 1.05
    
    sample_portfolio.update_from_transactions(new_timestamp, pd.DataFrame(), new_market_data)
    
    # Check stats after price increase
    stats_after_increase = sample_portfolio.get_stats()
    
    # Return should be higher after price increase
    assert stats_after_increase['total_return'] > stats_after_txn['total_return']
    assert stats_after_increase['total_return_pct'] > stats_after_txn['total_return_pct']
    
    # Number of trades should remain the same
    assert stats_after_increase['num_trades'] == stats_after_txn['num_trades']


def test_portfolio_value_calculation(sample_portfolio, sample_transactions, sample_market_data):
    """Test portfolio value calculation."""
    # Initial value
    initial_value = sample_portfolio.get_portfolio_value()
    assert initial_value == sample_portfolio.initial_capital
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Calculate expected portfolio value
    expected_position_value = (
        100 * 151.0 +  # AAPL
        50 * 251.0 +   # MSFT
        20 * 2510.0    # GOOG
    )
    expected_total = sample_portfolio.current_cash + expected_position_value
    
    # Check portfolio value
    portfolio_value = sample_portfolio.get_portfolio_value()
    assert portfolio_value == pytest.approx(expected_total)
    assert portfolio_value == pytest.approx(sample_portfolio.current_equity)


def test_get_positions(sample_portfolio, sample_transactions, sample_market_data):
    """Test getting positions as a DataFrame."""
    # Initial positions
    initial_positions = sample_portfolio.get_positions()
    assert initial_positions.empty
    
    # Update with transactions
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_portfolio.update_from_transactions(timestamp, sample_transactions, sample_market_data)
    
    # Get positions
    positions = sample_portfolio.get_positions()
    
    # Verify positions DataFrame
    assert not positions.empty
    assert len(positions) == 3  # Three symbols
    assert set(positions['symbol']) == {'AAPL', 'MSFT', 'GOOG'}
    assert set(positions.columns) == {'symbol', 'quantity', 'value'}
    
    # Verify position values
    aapl_row = positions[positions['symbol'] == 'AAPL'].iloc[0]
    assert aapl_row['quantity'] == 100
    assert aapl_row['value'] == 100 * 151.0
    
    msft_row = positions[positions['symbol'] == 'MSFT'].iloc[0]
    assert msft_row['quantity'] == 50
    assert msft_row['value'] == 50 * 251.0
    
    goog_row = positions[positions['symbol'] == 'GOOG'].iloc[0]
    assert goog_row['quantity'] == 20
    assert goog_row['value'] == 20 * 2510.0


def test_get_positions_history(sample_portfolio, sample_transactions, sample_market_data):
    """Test getting positions history."""
    # Initial positions history
    initial_history = sample_portfolio.get_positions_history()
    assert len(initial_history) == 0
    
    # Update with transactions at different timestamps
    timestamps = [
        datetime(2022, 1, 1, 9, 30),
        datetime(2022, 1, 1, 10, 30)
    ]
    
    # First transaction
    first_txn = pd.DataFrame([{
        'symbol': 'AAPL',
        'quantity': 100,
        'price': 150.0,
        'commission': 7.5
    }])
    
    first_market_data = pd.DataFrame([{
        'timestamp': timestamps[0],
        'symbol': 'AAPL',
        'open': 150.0,
        'high': 152.0,
        'low': 149.0,
        'close': 151.0,
        'volume': 1000000
    }])
    
    sample_portfolio.update_from_transactions(timestamps[0], first_txn, first_market_data)
    
    # Second transaction
    second_txn = pd.DataFrame([{
        'symbol': 'MSFT',
        'quantity': 50,
        'price': 250.0,
        'commission': 7.5
    }])
    
    second_market_data = pd.DataFrame([
        {
            'timestamp': timestamps[1],
            'symbol': 'AAPL',
            'open': 151.0,
            'high': 153.0,
            'low': 150.0,
            'close': 152.0,
            'volume': 1000000
        },
        {
            'timestamp': timestamps[1],
            'symbol': 'MSFT',
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        }
    ])
    
    sample_portfolio.update_from_transactions(timestamps[1], second_txn, second_market_data)
    
    # Get positions history
    positions_history = sample_portfolio.get_positions_history()
    
    # Verify positions history
    assert len(positions_history) == 2  # Two timestamps
    assert timestamps[0] in positions_history
    assert timestamps[1] in positions_history
    
    # Check first timestamp positions
    first_positions = positions_history[timestamps[0]]
    assert 'AAPL' in first_positions
    assert first_positions['AAPL']['quantity'] == 100
    assert first_positions['AAPL']['value'] == 100 * 151.0
    
    # Check second timestamp positions
    second_positions = positions_history[timestamps[1]]
    assert 'AAPL' in second_positions
    assert 'MSFT' in second_positions
    assert second_positions['AAPL']['quantity'] == 100
    assert second_positions['AAPL']['value'] == 100 * 152.0
    assert second_positions['MSFT']['quantity'] == 50
    assert second_positions['MSFT']['value'] == 50 * 251.0


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
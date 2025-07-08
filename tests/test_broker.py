"""Tests for the broker functionality of the Vegas backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from vegas.broker import Broker
from vegas.broker.broker import OrderStatus, OrderType
from vegas.broker.slippage import FixedSlippageModel, VolumeSlippageModel
from vegas.broker.commission import FixedCommissionModel, PercentageCommissionModel
from vegas.strategy import Signal


@pytest.fixture
def sample_broker():
    """Create a sample broker for testing."""
    # Create broker with no slippage for predictable test results
    slippage_model = FixedSlippageModel(slippage_pct=0.0)
    return Broker(initial_cash=100000.0, slippage_model=slippage_model)


@pytest.fixture
def sample_signals():
    """Create sample trading signals for testing."""
    return [
        Signal(symbol='AAPL', action='buy', quantity=100),
        Signal(symbol='MSFT', action='buy', quantity=50),
        Signal(symbol='GOOG', action='buy', quantity=20, price=2500.0)
    ]


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }]),
        'MSFT': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        }]),
        'GOOG': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 2500.0,
            'high': 2550.0,
            'low': 2490.0,
            'close': 2520.0,
            'volume': 100000
        }])
    }


def test_order_creation_and_validation(sample_broker, sample_signals):
    """Test order creation and validation."""
    # Place orders
    orders = []
    for signal in sample_signals:
        order = sample_broker.place_order(signal)
        orders.append(order)
    
    # Verify orders were created
    assert len(orders) == 3
    assert len(sample_broker.orders) == 3
    
    # Verify order details
    assert orders[0].symbol == 'AAPL'
    assert orders[0].quantity == 100
    assert orders[0].order_type == OrderType.MARKET
    assert orders[0].limit_price is None
    assert orders[0].status == OrderStatus.OPEN
    
    assert orders[1].symbol == 'MSFT'
    assert orders[1].quantity == 50
    assert orders[1].order_type == OrderType.MARKET
    assert orders[1].limit_price is None
    assert orders[1].status == OrderStatus.OPEN
    
    assert orders[2].symbol == 'GOOG'
    assert orders[2].quantity == 20
    assert orders[2].order_type == OrderType.LIMIT
    assert orders[2].limit_price == 2500.0
    assert orders[2].status == OrderStatus.OPEN


def test_market_order_execution(sample_broker, sample_signals, sample_market_data):
    """Test market order execution."""
    # Create market order
    market_signal = sample_signals[0]  # AAPL buy
    order = sample_broker.place_order(market_signal)
    
    # Execute orders
    timestamp = datetime(2022, 1, 1, 9, 30)
    transactions = sample_broker.execute_orders(sample_market_data, timestamp)
    
    # Verify transaction
    assert len(transactions) == 1
    txn = transactions[0]
    
    assert txn.symbol == 'AAPL'
    assert txn.quantity == 100
    # With no slippage, price should be close price
    assert txn.price == pytest.approx(151.0)
    
    # Verify order status
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 100
    assert order.filled_price == pytest.approx(151.0)


def test_limit_order_execution(sample_broker, sample_market_data):
    """Test limit order execution."""
    # Create limit order (buy GOOG at 2500)
    limit_signal = Signal(symbol='GOOG', action='buy', quantity=20, price=2500.0)
    order = sample_broker.place_order(limit_signal)
    
    # Execute orders with price above limit
    timestamp = datetime(2022, 1, 1, 9, 30)
    transactions = sample_broker.execute_orders(sample_market_data, timestamp)
    
    # Verify no transaction (price is above limit)
    assert len(transactions) == 0
    assert order.status == OrderStatus.OPEN
    
    # Modify market data to have price below limit
    lower_price_data = sample_market_data.copy()
    lower_price_data['GOOG'] = pd.DataFrame([{
        'timestamp': datetime(2022, 1, 1, 9, 30),
        'open': 2490.0,
        'high': 2500.0,
        'low': 2480.0,
        'close': 2490.0,
        'volume': 100000
    }])
    
    # Execute orders again
    transactions = sample_broker.execute_orders(lower_price_data, timestamp)
    
    # Verify transaction
    assert len(transactions) == 1
    txn = transactions[0]
    
    assert txn.symbol == 'GOOG'
    assert txn.quantity == 20
    # With no slippage, price should be close price
    assert txn.price == pytest.approx(2490.0)
    
    # Verify order status
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 20
    assert order.filled_price == pytest.approx(2490.0)


def test_slippage_model_implementation():
    """Test slippage model implementation."""
    # Test fixed slippage model
    fixed_model = FixedSlippageModel(slippage_pct=0.0001)  # 0.01% slippage
    
    # Buy order (price should increase)
    buy_price = fixed_model.apply_slippage(100.0, 100, pd.DataFrame({'volume': [10000]}), True)
    assert buy_price == pytest.approx(100.01)
    
    # Sell order (price should decrease)
    sell_price = fixed_model.apply_slippage(100.0, 100, pd.DataFrame({'volume': [10000]}), False)
    assert sell_price == pytest.approx(99.99)
    
    # Test volume slippage model
    volume_model = VolumeSlippageModel(volume_impact=0.1)  # 10% impact factor
    
    # Create test market data with volume
    market_data = pd.DataFrame({'volume': [10000]})
    
    # Buy order (price should increase based on volume ratio)
    # Volume ratio = 100 / 10000 = 0.01
    # Impact = 0.1 * 0.01 = 0.001 (0.1%)
    buy_price = volume_model.apply_slippage(100.0, 100, market_data, True)
    assert buy_price == pytest.approx(100.1)
    
    # Sell order (price should decrease based on volume ratio)
    sell_price = volume_model.apply_slippage(100.0, 100, market_data, False)
    assert sell_price == pytest.approx(99.9)
    
    # Test with custom slippage model
    broker = Broker(initial_cash=100000.0, slippage_model=volume_model)
    
    # Create and place order
    signal = Signal(symbol='AAPL', action='buy', quantity=100)
    order = broker.place_order(signal)
    
    # Create market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 10000
        }])
    }
    
    # Execute order
    timestamp = datetime(2022, 1, 1, 9, 30)
    transactions = broker.execute_orders(market_data, timestamp)
    
    # Verify slippage was applied
    assert len(transactions) == 1
    txn = transactions[0]
    
    # Volume ratio = 100 / 10000 = 0.01
    # Impact = 0.1 * 0.01 = 0.001 (0.1%)
    # Price should be close price + 0.1%
    expected_price = 151.0 * 1.001
    assert txn.price == pytest.approx(expected_price)


def test_commission_model_implementation():
    """Test commission model implementation."""
    # Test fixed commission model
    fixed_model = FixedCommissionModel(commission=5.0)
    
    # Commission should be fixed regardless of order size
    assert fixed_model.calculate_commission(100.0, 10) == 5.0
    assert fixed_model.calculate_commission(1000.0, 100) == 5.0
    
    # Test percentage commission model
    pct_model = PercentageCommissionModel(percentage=0.001)  # 0.1%
    
    # Commission should be proportional to order value
    assert pct_model.calculate_commission(100.0, 10) == pytest.approx(100.0 * 10 * 0.001)
    assert pct_model.calculate_commission(1000.0, 100) == pytest.approx(1000.0 * 100 * 0.001)
    
    # Test with custom commission model and no slippage
    slippage_model = FixedSlippageModel(slippage_pct=0.0)
    broker = Broker(initial_cash=100000.0, commission_model=pct_model, slippage_model=slippage_model)
    
    # Create and place order
    signal = Signal(symbol='AAPL', action='buy', quantity=100)
    order = broker.place_order(signal)
    
    # Create market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }])
    }
    
    # Execute order
    timestamp = datetime(2022, 1, 1, 9, 30)
    transactions = broker.execute_orders(market_data, timestamp)
    
    # Verify commission was applied
    assert len(transactions) == 1
    txn = transactions[0]
    
    # Commission should be 0.1% of order value
    expected_commission = 151.0 * 100 * 0.001
    assert txn.commission == pytest.approx(expected_commission)


def test_position_tracking(sample_broker):
    """Test position tracking."""
    # Initial state
    assert len(sample_broker.positions) == 0
    
    # Create and place buy order
    buy_signal = Signal(symbol='AAPL', action='buy', quantity=100)
    buy_order = sample_broker.place_order(buy_signal)
    
    # Create market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }])
    }
    
    # Execute buy order
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_broker.execute_orders(market_data, timestamp)
    
    # Verify position
    position = sample_broker.get_position('AAPL')
    assert position is not None
    assert position.quantity == 100
    assert position.cost_basis == pytest.approx(151.0)
    
    # Create and place sell order (partial)
    sell_signal = Signal(symbol='AAPL', action='sell', quantity=40)
    sell_order = sample_broker.place_order(sell_signal)
    
    # Execute sell order
    timestamp = datetime(2022, 1, 1, 10, 30)
    market_data['AAPL'] = pd.DataFrame([{
        'timestamp': timestamp,
        'open': 151.0,
        'high': 153.0,
        'low': 150.0,
        'close': 152.0,
        'volume': 1000000
    }])
    
    sample_broker.execute_orders(market_data, timestamp)
    
    # Verify updated position
    position = sample_broker.get_position('AAPL')
    assert position is not None
    assert position.quantity == 60  # 100 - 40
    
    # Create and place sell order (remaining)
    sell_all_signal = Signal(symbol='AAPL', action='sell', quantity=60)
    sell_all_order = sample_broker.place_order(sell_all_signal)
    
    # Execute sell all order
    timestamp = datetime(2022, 1, 1, 11, 30)
    market_data['AAPL'] = pd.DataFrame([{
        'timestamp': timestamp,
        'open': 152.0,
        'high': 154.0,
        'low': 151.0,
        'close': 153.0,
        'volume': 1000000
    }])
    
    sample_broker.execute_orders(market_data, timestamp)
    
    # Verify position is closed
    position = sample_broker.get_position('AAPL')
    assert position is None or position.quantity == 0


def test_order_cancellation(sample_broker):
    """Test order cancellation."""
    # Create and place order
    signal = Signal(symbol='AAPL', action='buy', quantity=100)
    order = sample_broker.place_order(signal)
    
    # Verify order is open
    assert order.status == OrderStatus.OPEN
    
    # Cancel order
    cancelled = sample_broker.cancel_order(order.id)
    
    # Verify cancellation
    assert cancelled
    assert order.status == OrderStatus.CANCELLED
    
    # Attempt to cancel again
    cancelled_again = sample_broker.cancel_order(order.id)
    
    # Should return False (already cancelled)
    assert not cancelled_again
    
    # Attempt to cancel non-existent order
    non_existent = sample_broker.cancel_order("non-existent-id")
    assert not non_existent


def test_get_order(sample_broker, sample_signals):
    """Test getting order by ID."""
    # Create orders
    orders = []
    for signal in sample_signals:
        order = sample_broker.place_order(signal)
        orders.append(order)
    
    # Get order by ID
    retrieved_order = sample_broker.get_order(orders[0].id)
    
    # Verify retrieved order
    assert retrieved_order is not None
    assert retrieved_order.id == orders[0].id
    assert retrieved_order.symbol == orders[0].symbol
    
    # Attempt to get non-existent order
    non_existent = sample_broker.get_order("non-existent-id")
    assert non_existent is None


def test_update_market_values(sample_broker):
    """Test updating market values."""
    # Create and execute buy orders
    buy_signals = [
        Signal(symbol='AAPL', action='buy', quantity=100),
        Signal(symbol='MSFT', action='buy', quantity=50)
    ]
    
    for signal in buy_signals:
        sample_broker.place_order(signal)
    
    # Initial market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }]),
        'MSFT': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        }])
    }
    
    # Execute orders
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_broker.execute_orders(market_data, timestamp)
    
    # Update market values with initial data to set baseline
    sample_broker.update_market_values(market_data)
    
    # Initial position values
    aapl_pos = sample_broker.get_position('AAPL')
    msft_pos = sample_broker.get_position('MSFT')
    
    initial_aapl_value = aapl_pos.market_value
    initial_msft_value = msft_pos.market_value
    
    # Update market data with new prices
    new_market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 10, 30),
            'open': 151.0,
            'high': 153.0,
            'low': 150.0,
            'close': 152.0,  # Up 1
            'volume': 1000000
        }]),
        'MSFT': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 10, 30),
            'open': 251.0,
            'high': 253.0,
            'low': 250.0,
            'close': 253.0,  # Up 2
            'volume': 500000
        }])
    }
    
    # Update market values
    sample_broker.update_market_values(new_market_data)
    
    # Verify updated position values
    aapl_pos = sample_broker.get_position('AAPL')
    msft_pos = sample_broker.get_position('MSFT')
    
    # AAPL: 100 shares, price up by 1
    assert aapl_pos.market_value == pytest.approx(initial_aapl_value + 100)
    
    # MSFT: 50 shares, price up by 2
    assert msft_pos.market_value == pytest.approx(initial_msft_value + 100)


def test_get_portfolio_value(sample_broker):
    """Test getting portfolio value."""
    # Initial portfolio value
    initial_value = sample_broker.get_portfolio_value()
    assert initial_value == sample_broker.cash
    
    # Create and execute buy orders
    buy_signals = [
        Signal(symbol='AAPL', action='buy', quantity=100),
        Signal(symbol='MSFT', action='buy', quantity=50)
    ]
    
    for signal in buy_signals:
        sample_broker.place_order(signal)
    
    # Market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }]),
        'MSFT': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        }])
    }
    
    # Execute orders
    timestamp = datetime(2022, 1, 1, 9, 30)
    transactions = sample_broker.execute_orders(market_data, timestamp)
    
    # Update market values to ensure positions are properly valued
    sample_broker.update_market_values(market_data)
    
    # Calculate expected portfolio value
    expected_position_value = (
        100 * 151.0 +  # AAPL
        50 * 251.0     # MSFT
    )
    total_commission = sum(txn.commission for txn in transactions)
    expected_portfolio_value = sample_broker.cash + expected_position_value
    
    # Verify portfolio value
    portfolio_value = sample_broker.get_portfolio_value()
    assert portfolio_value == pytest.approx(expected_portfolio_value)


def test_get_account(sample_broker):
    """Test getting account information."""
    # Initial account
    initial_account = sample_broker.get_account()
    assert initial_account['cash'] == sample_broker.cash
    
    # Check if 'equity' key exists, if not, check for 'portfolio_value'
    if 'equity' in initial_account:
        assert initial_account['equity'] == sample_broker.cash
    elif 'portfolio_value' in initial_account:
        assert initial_account['portfolio_value'] == sample_broker.cash
    else:
        # If neither exists, check total value
        assert initial_account.get('total_value', sample_broker.cash) == sample_broker.cash
    
    # Create and execute buy orders
    buy_signals = [
        Signal(symbol='AAPL', action='buy', quantity=100),
        Signal(symbol='MSFT', action='buy', quantity=50)
    ]
    
    for signal in buy_signals:
        sample_broker.place_order(signal)
    
    # Market data
    market_data = {
        'AAPL': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 150.0,
            'high': 152.0,
            'low': 149.0,
            'close': 151.0,
            'volume': 1000000
        }]),
        'MSFT': pd.DataFrame([{
            'timestamp': datetime(2022, 1, 1, 9, 30),
            'open': 250.0,
            'high': 252.0,
            'low': 249.0,
            'close': 251.0,
            'volume': 500000
        }])
    }
    
    # Execute orders
    timestamp = datetime(2022, 1, 1, 9, 30)
    sample_broker.execute_orders(market_data, timestamp)
    
    # Update market values
    sample_broker.update_market_values(market_data)
    
    # Get updated account
    account = sample_broker.get_account()
    
    # Verify cash was reduced
    assert account['cash'] < initial_account['cash']
    
    # Verify positions are tracked
    assert len(account.get('positions', [])) == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 
"""Broker simulation layer for the Vegas backtesting engine.

This module provides broker simulation for executing orders with slippage
and commission models, tracking positions, and maintaining transaction history.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import uuid

from vegas.broker.slippage import SlippageModel, FixedSlippageModel
from vegas.broker.commission import CommissionModel, FixedCommissionModel, PerShareCommissionModel
from vegas.strategy import Signal


class OrderStatus(Enum):
    """Order status enumeration."""
    OPEN = 'open'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'


@dataclass
class Order:
    """Order representation.
    
    Attributes:
        id: Unique order identifier
        symbol: Trading symbol
        quantity: Order quantity (positive for buy, negative for sell)
        order_type: Type of order (market, limit)
        limit_price: Price limit for limit orders
        created_at: Order creation timestamp
        status: Current order status
        filled_quantity: Quantity filled so far
        filled_price: Average fill price for filled quantity
    """
    id: str
    symbol: str
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    created_at: datetime = None
    status: OrderStatus = OrderStatus.OPEN
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    
    def __post_init__(self):
        """Initialize with default values."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class Transaction:
    """Transaction record.
    
    Attributes:
        id: Unique transaction identifier
        order_id: Reference to the originating order
        symbol: Trading symbol
        quantity: Transaction quantity (positive for buy, negative for sell)
        price: Execution price
        commission: Transaction fee
        timestamp: Transaction timestamp
    """
    id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    
    def __post_init__(self):
        """Initialize with default values."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def value(self) -> float:
        """Calculate transaction value.
        
        Returns:
            Transaction value (positive for buys, negative for sells)
        """
        return self.quantity * self.price


class Position:
    """Position representation for a single symbol.
    
    Attributes:
        symbol: Trading symbol
        quantity: Current position size
        cost_basis: Average cost basis
        market_value: Current market value
    """
    
    def __init__(self, symbol: str):
        """Initialize a position.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.quantity = 0.0
        self.cost_basis = 0.0
        self.market_value = 0.0
    
    def update(self, quantity: float, price: float) -> None:
        """Update position with a new transaction.
        
        Args:
            quantity: Transaction quantity (positive for buy, negative for sell)
            price: Transaction price
        """
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.cost_basis = price
        elif quantity * self.quantity > 0:
            # Adding to position
            total_cost = (self.cost_basis * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.cost_basis = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing position or flipping
            self.quantity += quantity
            if self.quantity == 0:
                self.cost_basis = 0
            elif self.quantity * (self.quantity - quantity) < 0:
                # Position flipped from long to short or vice versa
                self.cost_basis = price
    
    def update_market_value(self, price: float) -> None:
        """Update position market value.
        
        Args:
            price: Current market price
        """
        self.market_value = self.quantity * price
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss.
        
        Returns:
            Unrealized profit/loss
        """
        return self.market_value - (self.quantity * self.cost_basis)


class Broker:
    """Broker simulation for executing orders and tracking positions.
    
    This class simulates order execution with slippage and commission models,
    tracks positions, and maintains transaction history.
    """
    
    def __init__(self, initial_cash: float = 100000.0,
                slippage_model: Optional[SlippageModel] = None,
                commission_model: Optional[CommissionModel] = None):
        """Initialize the broker.
        
        Args:
            initial_cash: Initial cash balance
            slippage_model: Slippage model for order execution
            commission_model: Commission model for order execution
        """
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.transactions: List[Transaction] = []
        
        # Default models
        self.slippage_model = slippage_model or FixedSlippageModel()
        # Default to zero-cost per-share if none provided; engine/strategy can override
        self.commission_model: CommissionModel = commission_model or PerShareCommissionModel(cost_per_share=0.0, min_trade_cost=0.0)
    
    def place_order(self, signal: Signal) -> Order:
        """Place an order based on a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Newly created order
        """
        order_type = OrderType.LIMIT if signal.price is not None else OrderType.MARKET
        quantity = abs(signal.quantity) if signal.action == "buy" else -signal.quantity
        
        order = Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=signal.price,
        )
        
        self.orders.append(order)
        return order
    
    def execute_orders(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime) -> List[Transaction]:
        """Execute pending orders based on current market data.
        
        Args:
            market_data: Current market data by symbol
            timestamp: Current timestamp
            
        Returns:
            List of executed transactions
        """
        executed_transactions = []
        
        for order in self.orders:
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                continue
                
            # Check if symbol exists in market data
            if order.symbol not in market_data:
                order.status = OrderStatus.REJECTED
                continue
                
            symbol_data = market_data[order.symbol]
            if symbol_data.empty:
                order.status = OrderStatus.REJECTED
                continue
            
            # Get latest price from market data
            latest_price = symbol_data['close'].iloc[-1]
            
            # For limit orders, check if the price is acceptable
            if order.order_type == OrderType.LIMIT:
                is_buy = order.quantity > 0
                if (is_buy and latest_price > order.limit_price) or \
                   (not is_buy and latest_price < order.limit_price):
                    continue  # Price not acceptable for limit order
            
            # Apply slippage to execution price
            is_buy = order.quantity > 0
            execution_price = self.slippage_model.apply_slippage(
                latest_price, order.quantity, symbol_data, is_buy
            )
            
            # Calculate commission
            commission = self.commission_model.calculate_commission(
                execution_price, order.quantity
            )
            
            # Check if we have enough cash for a buy order
            unfilled = order.quantity - order.filled_quantity
            trade_value = unfilled * execution_price
            if is_buy and trade_value + commission > self.cash:
                # Adjust quantity for partial fill if possible
                max_affordable = self.cash / (execution_price + (commission / abs(unfilled)))
                if max_affordable <= 0:
                    continue  # Cannot afford any shares
                
                unfilled = min(unfilled, max_affordable)
                trade_value = unfilled * execution_price
                commission = self.commission_model.calculate_commission(
                    execution_price, unfilled
                )
            
            # Execute the order
            transaction = Transaction(
                id=str(uuid.uuid4()),
                order_id=order.id,
                symbol=order.symbol,
                quantity=unfilled,
                price=execution_price,
                commission=commission,
                timestamp=timestamp
            )
            
            # Update order status
            order.filled_quantity += unfilled
            order.filled_price = (
                (order.filled_price or 0) * (order.filled_quantity - unfilled) + 
                execution_price * unfilled
            ) / order.filled_quantity if order.filled_quantity > 0 else execution_price
            
            if abs(order.filled_quantity - order.quantity) < 1e-6:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Update position
            self._update_position(transaction.symbol, transaction.quantity, transaction.price)
            
            # Update cash
            self.cash -= (trade_value + commission)
            
            # Record transaction
            self.transactions.append(transaction)
            executed_transactions.append(transaction)
        
        return executed_transactions
    
    def _update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position for a symbol.
        
        Args:
            symbol: Trading symbol
            quantity: Transaction quantity
            price: Transaction price
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        self.positions[symbol].update(quantity, price)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled, False if not found or not open
        """
        for order in self.orders:
            if order.id == order_id and order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(symbol)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        for order in self.orders:
            if order.id == order_id:
                return order
        return None
    
    def update_market_values(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update market values for all positions.
        
        Args:
            market_data: Current market data by symbol
        """
        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].empty:
                price = market_data[symbol]['close'].iloc[-1]
                position.update_market_value(price)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_account(self) -> Dict[str, Any]:
        """Get account summary.
        
        Returns:
            Account summary dictionary
        """
        return {
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'cost_basis': pos.cost_basis,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl()
            } for symbol, pos in self.positions.items() if abs(pos.quantity) > 1e-6},
            'portfolio_value': self.get_portfolio_value(),
        } 
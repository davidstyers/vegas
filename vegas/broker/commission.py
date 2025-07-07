"""Commission models for the Vegas backtesting engine.

This module provides commission models for simulating trading fees
during order execution in the broker simulation layer.
"""

from abc import ABC, abstractmethod


class CommissionModel(ABC):
    """Abstract base class for commission models.
    
    Commission models calculate trading fees for order execution.
    """
    
    @abstractmethod
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate commission for an order.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        pass


class FixedCommissionModel(CommissionModel):
    """Fixed commission model.
    
    This model charges a fixed fee per trade, regardless of size.
    """
    
    def __init__(self, commission: float = 5.0):
        """Initialize the fixed commission model.
        
        Args:
            commission: Fixed commission amount per trade (default: $5.00)
        """
        self.commission = commission
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate fixed commission for an order.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        return self.commission if quantity != 0 else 0.0


class PerShareCommissionModel(CommissionModel):
    """Per-share commission model.
    
    This model charges a fee per share traded, with an optional minimum.
    """
    
    def __init__(self, cost_per_share: float = 0.005, min_trade_cost: float = 1.0):
        """Initialize the per-share commission model.
        
        Args:
            cost_per_share: Commission per share (default: $0.005)
            min_trade_cost: Minimum commission per trade (default: $1.00)
        """
        self.cost_per_share = cost_per_share
        self.min_trade_cost = min_trade_cost
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate per-share commission for an order.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        if quantity == 0:
            return 0.0
        
        commission = abs(quantity) * self.cost_per_share
        return max(commission, self.min_trade_cost)


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission model.
    
    This model charges a percentage of the trade value, with optional minimum and maximum.
    """
    
    def __init__(self, percentage: float = 0.001, min_trade_cost: float = 1.0, 
                max_trade_cost: float = float('inf')):
        """Initialize the percentage commission model.
        
        Args:
            percentage: Commission percentage (default: 0.1%)
            min_trade_cost: Minimum commission per trade (default: $1.00)
            max_trade_cost: Maximum commission per trade (default: unlimited)
        """
        self.percentage = percentage
        self.min_trade_cost = min_trade_cost
        self.max_trade_cost = max_trade_cost
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate percentage-based commission for an order.
        
        Args:
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        if quantity == 0:
            return 0.0
        
        trade_value = abs(price * quantity)
        commission = trade_value * self.percentage
        
        # Apply minimum and maximum constraints
        commission = max(commission, self.min_trade_cost)
        commission = min(commission, self.max_trade_cost)
        
        return commission 
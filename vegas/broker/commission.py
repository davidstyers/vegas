"""Commission models for the Vegas backtesting engine.

This module provides commission models for simulating trading fees
during order execution in the broker simulation layer.
"""

from abc import ABC, abstractmethod


class CommissionModel(ABC):
    """Abstract base class for commission models."""
    @abstractmethod
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate commission for an order."""
        raise NotImplementedError


class FixedCommissionModel(CommissionModel):
    """Fixed commission model (fixed fee per trade)."""
    def __init__(self, commission: float = 5.0):
        self.commission = commission
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        return self.commission if quantity != 0 else 0.0


class PerShareCommissionModel(CommissionModel):
    """Per-share commission model with optional minimum trade cost."""
    def __init__(self, cost_per_share: float = 0.005, min_trade_cost: float = 1.0):
        self.cost_per_share = cost_per_share
        self.min_trade_cost = min_trade_cost
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        if quantity == 0:
            return 0.0
        commission = abs(quantity) * self.cost_per_share
        return max(commission, self.min_trade_cost)


class PercentageCommissionModel(CommissionModel):
    """Percentage-based commission model with min/max bounds."""
    def __init__(self, percentage: float = 0.001, min_trade_cost: float = 1.0,
                 max_trade_cost: float = float('inf')):
        self.percentage = percentage
        self.min_trade_cost = min_trade_cost
        self.max_trade_cost = max_trade_cost
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        if quantity == 0:
            return 0.0
        trade_value = abs(price * quantity)
        commission = trade_value * self.percentage
        commission = max(commission, self.min_trade_cost)
        commission = min(commission, self.max_trade_cost)
        return commission


# Zipline-like helper namespace for strategies
class commission:
    """Helper to allow strategy usage: context.set_commission(commission.PerShare(...))."""
    @staticmethod
    def Fixed(commission: float = 5.0) -> CommissionModel:
        return FixedCommissionModel(commission=commission)

    @staticmethod
    def PerShare(cost_per_share: float = 0.005, min_trade_cost: float = 1.0) -> CommissionModel:
        return PerShareCommissionModel(cost_per_share=cost_per_share, min_trade_cost=min_trade_cost)

    @staticmethod
    def PerTrade(percent: float = 0.001, min_trade_cost: float = 1.0, max_trade_cost: float = float('inf')) -> CommissionModel:
        return PercentageCommissionModel(percentage=percent, min_trade_cost=min_trade_cost, max_trade_cost=max_trade_cost)
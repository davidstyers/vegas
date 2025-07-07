"""Slippage models for the Vegas backtesting engine.

This module provides slippage models for simulating price impact
during order execution in the broker simulation layer.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class SlippageModel(ABC):
    """Abstract base class for slippage models.
    
    Slippage models adjust execution prices to simulate market impact.
    """
    
    @abstractmethod
    def apply_slippage(self, order_price: float, order_quantity: float, 
                      market_data: pd.DataFrame, is_buy: bool) -> float:
        """Apply slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed percentage slippage model.
    
    This model adjusts execution prices by a fixed percentage.
    """
    
    def __init__(self, slippage_pct: float = 0.001):
        """Initialize the fixed slippage model.
        
        Args:
            slippage_pct: Slippage percentage (default: 0.001%)
        """
        self.slippage_pct = slippage_pct
    
    def apply_slippage(self, order_price: float, order_quantity: float,
                      market_data: pd.DataFrame, is_buy: bool) -> float:
        """Apply fixed percentage slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        # For buys: price increases, for sells: price decreases
        direction = 1 if is_buy else -1
        slippage_factor = 1 + (direction * self.slippage_pct)
        return order_price * slippage_factor


class VolumeSlippageModel(SlippageModel):
    """Volume-based slippage model.
    
    This model adjusts execution prices based on order size relative to volume.
    """
    
    def __init__(self, volume_impact: float = 0.1):
        """Initialize the volume slippage model.
        
        Args:
            volume_impact: Impact factor for volume-based slippage (default: 0.1)
        """
        self.volume_impact = volume_impact
    
    def apply_slippage(self, order_price: float, order_quantity: float,
                      market_data: pd.DataFrame, is_buy: bool) -> float:
        """Apply volume-based slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        # Get current volume from market data
        volume = market_data['volume'].iloc[-1]
        
        # Calculate volume ratio (order size / market volume)
        volume_ratio = min(abs(order_quantity) / max(volume, 1), 1.0)
        
        # Calculate price impact based on volume ratio and impact factor
        impact = self.volume_impact * volume_ratio
        
        # For buys: price increases, for sells: price decreases
        direction = 1 if is_buy else -1
        slippage_factor = 1 + (direction * impact)
        
        return order_price * slippage_factor 
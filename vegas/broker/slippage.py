"""Slippage models for the Vegas backtesting engine.

This module provides slippage models for simulating price impact
during order execution in the broker simulation layer.

Polars-only implementation; pandas is not used.
"""

from abc import ABC, abstractmethod
import polars as pl


class SlippageModel(ABC):
    """Abstract base class for slippage models.
    
    Slippage models adjust execution prices to simulate market impact.
    """
    
    @abstractmethod
    def apply_slippage(self, order_price: float, order_quantity: float,
                      market_data: pl.DataFrame, is_buy: bool) -> float:
        """Apply slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data (Polars DataFrame)
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        raise NotImplementedError


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
                      market_data: pl.DataFrame, is_buy: bool) -> float:
        """Apply fixed percentage slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data (Polars DataFrame)
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        # For buys: price increases, for sells: price decreases
        direction = 1.0 if is_buy else -1.0
        slippage_factor = 1.0 + (direction * float(self.slippage_pct))
        return float(order_price) * slippage_factor


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
                      market_data: pl.DataFrame, is_buy: bool) -> float:
        """Apply volume-based slippage to an order.
        
        Args:
            order_price: Intended execution price
            order_quantity: Order quantity
            market_data: Current market data (Polars DataFrame)
            is_buy: Whether the order is a buy (True) or sell (False)
            
        Returns:
            Adjusted execution price after slippage
        """
        # Get current volume from market data (last row)
        if market_data.is_empty() or 'volume' not in market_data.columns:
            return float(order_price)
        volume = float(market_data.item(-1, 'volume'))
        
        # Calculate volume ratio (order size / market volume)
        volume = max(volume, 1.0)
        volume_ratio = min(abs(float(order_quantity)) / volume, 1.0)
        
        # Calculate price impact based on volume ratio and impact factor
        impact = float(self.volume_impact) * volume_ratio
        
        # For buys: price increases, for sells: price decreases
        direction = 1.0 if is_buy else -1.0
        slippage_factor = 1.0 + (direction * impact)
        
        return float(order_price) * slippage_factor
"""Base classes for data transformation and resampling."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union
import polars as pl


class DataType(Enum):
    """Supported data types for transformation."""
    OHLCV = "ohlcv"
    TICK = "tick" 
    TBBO = "tbbo"


class FrequencyType(Enum):
    """Types of frequency/bar transformations."""
    TIME = "time"           # Time-based bars (1min, 5min, 1h, 1d)
    TICK = "tick"           # Tick bars
    VOLUME = "volume"       # Volume bars
    DOLLAR = "dollar"       # Dollar bars
    TICK_IMBALANCE = "tick_imbalance"         # Tick imbalance bars
    VOLUME_IMBALANCE = "volume_imbalance"     # Volume imbalance bars
    DOLLAR_IMBALANCE = "dollar_imbalance"     # Dollar imbalance bars
    TICK_RUN = "tick_run"                     # Tick run bars
    VOLUME_RUN = "volume_run"                 # Volume run bars
    DOLLAR_RUN = "dollar_run"                 # Dollar run bars


class DataTransformer(ABC):
    """Abstract base class for all data transformers."""
    
    def __init__(self, **kwargs):
        """Initialize transformer with configuration parameters."""
        self.config = kwargs
        
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform input DataFrame according to the specific method.
        
        Args:
            df: Input DataFrame with tick or OHLCV data
            
        Returns:
            Transformed DataFrame with new bar structure
        """
        raise NotImplementedError
        
    @abstractmethod
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate that input DataFrame has required columns and format.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        raise NotImplementedError
        
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        return []
        
    def get_output_schema(self) -> Dict[str, pl.DataType]:
        """Get schema for output DataFrame.
        
        Returns:
            Dictionary mapping column names to Polars data types
        """
        return {
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64, 
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }


class BarTransformer(DataTransformer):
    """Base class for bar-based transformers (tick, volume, dollar bars)."""
    
    def __init__(self, bar_size: Union[int, float], **kwargs):
        """Initialize bar transformer with bar size.
        
        Args:
            bar_size: Size threshold for creating new bars
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.bar_size = bar_size
        
    def _create_ohlcv_from_ticks(self, tick_group: pl.DataFrame) -> Dict:
        """Create OHLCV bar from group of ticks.
        
        Args:
            tick_group: DataFrame containing ticks for one bar
            
        Returns:
            Dictionary with OHLCV values
        """
        if tick_group.is_empty():
            return {
                "timestamp": None,
                "open": None,
                "high": None,
                "low": None, 
                "close": None,
                "volume": 0.0,
            }
            
        return {
            "timestamp": tick_group["timestamp"].min(),
            "open": tick_group["price"].first(),
            "high": tick_group["price"].max(),
            "low": tick_group["price"].min(),
            "close": tick_group["price"].last(),
            "volume": tick_group["volume"].sum() if "volume" in tick_group.columns else tick_group.height,
        }


class ImbalanceBarTransformer(BarTransformer):
    """Base class for imbalance bar transformers."""
    
    def __init__(self, bar_size: Union[int, float], expected_imbalance: float = 0.0, **kwargs):
        """Initialize imbalance bar transformer.
        
        Args:
            bar_size: Size threshold for creating new bars
            expected_imbalance: Expected imbalance value (default 0.0)
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        self.expected_imbalance = expected_imbalance
        
    def _calculate_tick_rule(self, prices: pl.Series) -> pl.Series:
        """Calculate tick rule (buy/sell classification) based on price changes.
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of tick rules (+1 for uptick, -1 for downtick, 0 for no change)
        """
        price_diff = prices.diff()
        tick_rule = pl.when(price_diff > 0).then(1)\
                     .when(price_diff < 0).then(-1)\
                     .otherwise(0)
        
        # Forward fill zero values with previous non-zero value
        return tick_rule.fill_null(strategy="forward").fill_null(0)


class RunBarTransformer(BarTransformer): 
    """Base class for run bar transformers."""
    
    def __init__(self, bar_size: Union[int, float], **kwargs):
        """Initialize run bar transformer.
        
        Args:
            bar_size: Size threshold for creating new bars
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def _calculate_runs(self, tick_rule: pl.Series) -> pl.Series:
        """Calculate runs of consecutive buy or sell ticks.
        
        Args:
            tick_rule: Series of tick rules (+1/-1)
            
        Returns:
            Series indicating run groups
        """
        # Create run groups by identifying changes in sign
        sign_changes = (tick_rule != tick_rule.shift(1)).cum_sum()
        return sign_changes

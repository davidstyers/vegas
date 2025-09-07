"""Data transformation and resampling module for Vegas backtesting engine.

This module provides comprehensive data transformation capabilities including:
- OHLCV resampling (1h -> 4h -> 1day, etc.)
- Tick bar transformations (Lopez de Prado methods)
- Imbalance bars (tick, volume, dollar)
- Run bars (tick, volume, dollar)

All implementations use Polars for maximum efficiency.
"""

from vegas.data.transform.ohlcv_resample import OHLCVResampler
from vegas.data.transform.tick_bars import (
    TickBars,
    VolumeBars, 
    DollarBars,
    TickImbalanceBars,
    VolumeImbalanceBars,
    DollarImbalanceBars,
    TickRunBars,
    VolumeRunBars,
    DollarRunBars,
)
from vegas.data.transform.base import DataTransformer
from vegas.data.transform.frequency_manager import FrequencyManager

__all__ = [
    "DataTransformer",
    "FrequencyManager", 
    "OHLCVResampler",
    "TickBars",
    "VolumeBars",
    "DollarBars", 
    "TickImbalanceBars",
    "VolumeImbalanceBars",
    "DollarImbalanceBars",
    "TickRunBars",
    "VolumeRunBars",
    "DollarRunBars",
]

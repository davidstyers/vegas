"""Factors for the Vegas pipeline system."""

from vegas.pipeline.factors.custom import CustomFactor
from vegas.pipeline.factors.basic import (
    Returns,
    SimpleMovingAverage,
    ExponentialWeightedMovingAverage,
    VWAP,
    StandardDeviation,
)
from vegas.pipeline.factors.statistical import ZScore, Rank, Percentile

__all__ = [
    'CustomFactor',
    'Returns',
    'SimpleMovingAverage',
    'ExponentialWeightedMovingAverage',
    'VWAP',
    'StandardDeviation',
    'ZScore',
    'Rank',
    'Percentile',
]

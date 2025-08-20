"""Utility functions and classes for the Vegas backtesting engine."""

# Explicit re-export for public API
from vegas.utils.data_generator import (
    generate_synthetic_data as generate_random_ohlcv_data,
)
from vegas.utils.logging import configure_logging

__all__ = [
    "generate_random_ohlcv_data",
    "configure_logging",
]

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, datetime
from typing import Iterable
import polars as pl


@dataclass
class TradingCalendar:
    """Resolution-agnostic trading calendar operating on precise timestamps.

    Subclasses should implement `is_trading_time` for a single datetime and
    may override `filter_timestamps` for vectorized performance.
    """

    name: str

    def is_trading_time(self, dt: datetime) -> bool:  # pragma: no cover - base fallback
        return True

    def filter_timestamps(self, timestamps: pl.Series) -> pl.Series:
        """Filter a pl.Series[Datetime] to those within valid trading periods.

        Args:
            timestamps: Polars Series of timezone-aware datetimes.

        Returns:
            Polars Series of the same dtype, filtered and sorted.
        """
        if timestamps.is_empty():
            return timestamps

        # Fallback generic filter via map if subclass didn't override
        mask = timestamps.map_elements(self.is_trading_time, return_dtype=pl.Boolean)
        return timestamps.filter(mask).sort()


@dataclass
class TwentyFourSevenCalendar(TradingCalendar):
    """Pass-through calendar for 24/7 markets (e.g., crypto)."""

    name: str = "24/7"

    def is_trading_time(self, dt: datetime) -> bool:
        return True

 
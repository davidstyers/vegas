from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass
class TradingCalendar:
    """Resolution-agnostic trading calendar for timestamp filtering.

    Subclasses implement `is_trading_time(dt)` and may override
    `filter_timestamps(series)` for vectorized performance.

    Each calendar declares a `timezone` string (IANA identifier) describing
    the local market timezone used for timestamp normalization.
    """

    name: str
    timezone: str

    def is_trading_time(self, dt: datetime) -> bool:  # pragma: no cover - base fallback
        return True

    def filter_timestamps(self, timestamps: pl.Series) -> pl.Series:
        """Return a filtered Series of timestamps within valid trading periods.

        :param timestamps: Polars Series of timezone-aware datetimes.
        :type timestamps: pl.Series
        :returns: Filtered and sorted Series of the same dtype.
        :rtype: pl.Series
        """
        if timestamps.is_empty():
            return timestamps

        # Fallback generic filter via map if subclass didn't override
        mask = timestamps.map_elements(self.is_trading_time, return_dtype=pl.Boolean)
        return timestamps.filter(mask).sort()


@dataclass
class TwentyFourSevenCalendar(TradingCalendar):
    """Pass-through calendar for 24/7 markets (e.g., crypto).

    Uses UTC as the canonical timezone.
    """

    name: str = "24/7"
    timezone: str = "UTC"

    def is_trading_time(self, dt: datetime) -> bool:
        return True

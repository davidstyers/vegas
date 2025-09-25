from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import polars as pl


@dataclass
class TradingCalendar:
    """Resolution-agnostic trading calendar for timestamp filtering.

    Subclasses implement `is_trading_time(dt)` and may override
    `filter_timestamps(series)` for vectorized performance.

    Each calendar declares a `timezone` string (IANA identifier) describing
    the local market timezone used for timestamp normalization.
    
    Market hours are defined by `market_open_time` and `market_close_time`
    which specify the local market open/close times. If None, the calendar
    has no specific market hours (e.g., 24/7 markets).
    """

    name: str
    timezone: str
    market_open_time: Optional[time] = None
    market_close_time: Optional[time] = None

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

    def get_market_hours_for_date(self, trading_date) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get market open and close times for a specific trading date.
        
        :param trading_date: The trading date (date object or datetime)
        :returns: Tuple of (market_open_datetime, market_close_datetime) or (None, None) if no market hours
        :raises ValueError: If only one of market_open_time/market_close_time is defined
        """
        if self.market_open_time is None and self.market_close_time is None:
            # No specific market hours (e.g., 24/7 markets)
            return None, None
            
        if self.market_open_time is None or self.market_close_time is None:
            raise ValueError(
                f"Calendar '{self.name}' has incomplete market hours definition. "
                f"Both market_open_time and market_close_time must be defined or both must be None."
            )
        
        try:
            # Convert date to datetime if needed
            if hasattr(trading_date, 'date'):
                trading_date = trading_date.date()
                
            # Create timezone-aware datetime objects
            open_time = datetime.combine(trading_date, self.market_open_time)
            close_time = datetime.combine(trading_date, self.market_close_time)
            
            # Convert to the calendar's timezone
            import pytz
            tz = pytz.timezone(self.timezone)
            open_time = tz.localize(open_time)
            close_time = tz.localize(close_time)
            
            return open_time, close_time
            
        except Exception as e:
            raise ValueError(f"Failed to get market hours for {trading_date}: {e}")


@dataclass
class TwentyFourSevenCalendar(TradingCalendar):
    """Pass-through calendar for 24/7 markets (e.g., crypto).

    Uses UTC as the canonical timezone.
    """

    name: str = "24/7"
    timezone: str = "UTC"

    def is_trading_time(self, dt: datetime) -> bool:
        return True

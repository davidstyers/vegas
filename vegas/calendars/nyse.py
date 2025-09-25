from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import polars as pl

from .base import TradingCalendar


@dataclass
class NYSECalendar(TradingCalendar):
    """NYSE-like calendar: Mon–Fri, 09:30–16:00 US/Eastern (no holidays).

    Note: Holidays are not modeled due to data driven approach. For production, extend with an observed
    holiday list and early-closes if required by your use case.
    """
    name: str = "NYSE"
    timezone: str = "US/Eastern"
    market_open_time: time = time(9, 30)  # 9:30 AM
    market_close_time: time = time(16, 0)  # 4:00 PM

    def is_trading_time(
        self, dt: datetime
    ) -> bool:  # pragma: no cover - vectorized path used
        weekday = dt.weekday()  # 0=Mon..6=Sun
        if weekday >= 5:
            return False
        minutes = dt.hour * 60 + dt.minute
        return 570 <= minutes < 960  # 09:30-16:00

    def filter_timestamps(self, timestamps: pl.Series) -> pl.Series:
        """Vectorized filtering for trading timestamps within NYSE hours.

        :param timestamps: Series of datetimes to filter.
        :type timestamps: pl.Series
        :returns: Filtered and sorted Series of datetimes.
        :rtype: pl.Series
        """
        if timestamps.is_empty():
            return timestamps
        s = timestamps
        # Weekday filter Mon-Fri
        wd = s.dt.weekday()
        # Minute-of-day
        minutes = (s.dt.hour().cast(pl.Int32) * 60) + s.dt.minute().cast(pl.Int32)
        mask = (wd < 5) & minutes.is_between(570, 960, closed="left")
        return s.filter(mask).sort()


@dataclass
class NYSECalendarExtended(TradingCalendar):
    """NYSE calendar with extended hours (pre-market and after-hours).
    
    Extended trading hours:
    - Pre-market: 4:00 AM - 9:30 AM ET
    - Regular hours: 9:30 AM - 4:00 PM ET
    - After-hours: 4:00 PM - 8:00 PM ET
    
    Note: market_open_time and market_close_time still refer to regular trading hours
    for the on_market_open and on_market_close strategy hooks.
    """

    name: str = "NYSE_EXT"
    timezone: str = "US/Eastern"
    market_open_time: time = time(9, 30)  # Regular market open
    market_close_time: time = time(16, 0)  # Regular market close

    def is_trading_time(self, dt: datetime) -> bool:
        """Check if timestamp falls within extended trading hours (4 AM - 8 PM ET)."""
        weekday = dt.weekday()  # 0=Mon..6=Sun
        if weekday >= 5:  # Weekend
            return False
        
        minutes = dt.hour * 60 + dt.minute
        # Extended hours: 4:00 AM (240 min) to 8:00 PM (1200 min)
        return 240 <= minutes < 1200

    def filter_timestamps(self, timestamps: pl.Series) -> pl.Series:
        """Vectorized filtering for extended trading hours (4 AM - 8 PM ET)."""
        if timestamps.is_empty():
            return timestamps
        
        s = timestamps
        # Weekday filter Mon-Fri
        wd = s.dt.weekday()
        # Minute-of-day for extended hours
        minutes = (s.dt.hour().cast(pl.Int32) * 60) + s.dt.minute().cast(pl.Int32)
        # Extended hours: 4:00 AM (240) to 8:00 PM (1200)
        mask = (wd < 5) & minutes.is_between(240, 1200, closed="left")
        return s.filter(mask).sort()
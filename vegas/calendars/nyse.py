from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import polars as pl

from .base import TradingCalendar


@dataclass
class NYSECalendar(TradingCalendar):
    """NYSE-like calendar: Mon–Fri, 09:30–16:00 US/Eastern (no holidays).

    Note: Holidays are not modeled. For production, extend with an observed
    holiday list and early-closes if required by your use case.
    """

    name: str = "NYSE"

    def is_trading_time(self, dt: datetime) -> bool:  # pragma: no cover - vectorized path used
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


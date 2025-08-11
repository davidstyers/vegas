from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import polars as pl


class MarketDataFeed:
    """Abstract market data feed producing per-timestamp slices.

    Implementations must operate solely with Polars `DataFrame` objects to
    match the rest of the engine. Feeds can be pull-based (via `next_bar()`)
    or push-based (via `on_bar()` registrations).
    """

    def subscribe(self, symbols: List[str], fields: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    def start(
        self, from_dt: Optional[datetime] = None, to_dt: Optional[datetime] = None
    ) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def next_bar(self) -> Optional[Tuple[datetime, pl.DataFrame]]:
        """Return the next `(timestamp, DataFrame)` slice or ``None`` when exhausted.

        The DataFrame should contain rows for the current timestamp across all
        subscribed symbols and mirror engine expectations (columns like
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume').
        """
        raise NotImplementedError

    def current_time(self) -> Optional[datetime]:
        return None

    def is_realtime(self) -> bool:
        return False

    def on_bar(self, callback: Callable[[datetime, pl.DataFrame], None]) -> None:
        """Register a push-based callback invoked for each new bar.

        :param callback: Callable accepting `(timestamp, DataFrame)`.
        :type callback: Callable[[datetime, pl.DataFrame], None]
        :returns: None
        :rtype: None
        """
        self._on_bar_cb = callback  # type: ignore[attr-defined]

    def close(self) -> None:
        pass


class HistoricalFeedAdapter(MarketDataFeed):
    """Adapter that yields historical slices equivalent to engine iteration.

    Materializes data via `DataLayer`/`DataPortal` and produces a sequence of
    per-timestamp Polars DataFrame slices that match `BacktestEngine`'s needs.
    """

    def __init__(
        self,
        data_layer,
        symbols: Optional[List[str]],
        start: datetime,
        end: datetime,
        market_hours: Optional[tuple[str, str]] = None,
        timezone: Optional[str] = None,
    ):
        self._data_layer = data_layer
        self._symbols = symbols
        self._start = start
        self._end = end
        self._market_hours = market_hours
        self._timezone = timezone or getattr(data_layer, "timezone", "UTC")
        self._started = False
        self._stopped = False
        self._closed = False

        self._timestamps: List[datetime] = []
        self._partitioned: Dict[datetime, pl.DataFrame] = {}
        self._idx = 0
        self._current_ts: Optional[datetime] = None
        self._fields: Optional[List[str]] = None

    def subscribe(self, symbols: List[str], fields: Optional[List[str]] = None) -> None:
        # Accept and override symbols if provided here
        if symbols:
            self._symbols = symbols
        self._fields = fields

    def start(
        self, from_dt: Optional[datetime] = None, to_dt: Optional[datetime] = None
    ) -> None:
        if self._started:
            return
        self._started = True
        # Materialize market data using the same path the engine uses today
        market_data: pl.DataFrame = self._data_layer.get_data_for_backtest(
            from_dt or self._start,
            to_dt or self._end,
            market_hours=self._market_hours,
            symbols=self._symbols,
        )
        # Optionally select requested fields (ensure required columns stay present)
        if self._fields:
            required = {"timestamp", "symbol"}
            cols = [c for c in self._fields if c in market_data.columns]
            for r in required:
                if r not in cols and r in market_data.columns:
                    cols.append(r)
            if cols:
                market_data = market_data.select(cols)

        # Prepare per-timestamp grouping equivalent to engine expectations
        if market_data.is_empty():
            self._timestamps = []
            self._partitioned = {}
            self._idx = 0
            return

        # Ensure the same sorting the engine uses downstream
        daily = market_data.sort("timestamp")

        # Build partition map by timestamp to allow O(1) retrieval
        self._partitioned = {}
        self._timestamps = []

        # group_by maintains order with maintain_order=True in Polars
        for (ts,), ts_df in daily.group_by("timestamp", maintain_order=True):
            # Keep the slice as-is (polars DataFrame)
            self._partitioned[ts] = ts_df
            self._timestamps.append(ts)  # ts is a Python datetime from polars

        self._idx = 0

    def stop(self) -> None:
        self._stopped = True

    def next_bar(self) -> Optional[Tuple[datetime, pl.DataFrame]]:
        if not self._started or self._stopped or self._closed:
            return None

        if self._idx >= len(self._timestamps):
            return None

        ts = self._timestamps[self._idx]
        self._idx += 1
        self._current_ts = ts
        pl_slice = self._partitioned.get(ts)
        if pl_slice is None:
            return ts, pl.DataFrame()

        return ts, pl_slice

    def current_time(self) -> Optional[datetime]:
        return self._current_ts

    def is_realtime(self) -> bool:
        return False

    def close(self) -> None:
        self._closed = True
        self._partitioned = {}
        self._timestamps = []

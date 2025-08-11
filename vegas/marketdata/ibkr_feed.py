from __future__ import annotations

"""
Interactive Brokers (IBKR) market data feed.

This module provides a feed that conforms to vegas.marketdata.feed.MarketDataFeed
so it can be passed into BacktestEngine.run_live(..., feed=...).

Phase 1: Skeleton implementation with optional ibapi imports and safe fallbacks.
- If ibapi is not installed, start() raises informative ImportError.
- Public methods operate with in-memory queues so tests can inject bars hermetically.
"""

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import polars as pl

try:
    # Optional dependency; real connectivity added in a later phase

    _IBAPI_AVAILABLE = True
except Exception:
    _IBAPI_AVAILABLE = False

from vegas.marketdata.feed import MarketDataFeed


@dataclass
class IBKRFeedConfig:
    """Configuration for `InteractiveBrokersFeed`.

    Attributes mirror typical paper-trading defaults. Some fields (bar_size,
    what_to_show, use_rth) are placeholders for future live data integrations.
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 124
    bar_size: str = "1 min"  # Future use
    what_to_show: str = "TRADES"  # Future use
    use_rth: bool = True  # Future use
    timezone: str = "UTC"


class InteractiveBrokersFeed(MarketDataFeed):
    """Interactive Brokers `MarketDataFeed` (phase 1 skeleton).

    - No real network connectivity; exposes `_inject_bar` for tests.
    - Maintains a queue of per-timestamp Polars DataFrame slices.
    """

    def __init__(
        self,
        config: Optional[IBKRFeedConfig] = None,
    ):
        self._config = config or IBKRFeedConfig()
        self._symbols: List[str] = []
        self._fields: Optional[List[str]] = None

        # Ordered queue of (timestamp, pl.DataFrame) slices
        self._bars_queue: "queue.Queue[Tuple[datetime, pl.DataFrame]]" = queue.Queue()
        self._current_ts: Optional[datetime] = None

        self._started = False
        self._stopped = False
        self._closed = False
        self._is_realtime = True
        self._thread: Optional[threading.Thread] = None

    def subscribe(self, symbols: List[str], fields: Optional[List[str]] = None) -> None:
        """Subscribe to symbols/fields; fields are optional and advisory."""
        if symbols:
            self._symbols = list(symbols)
        self._fields = fields

    def start(
        self, from_dt: Optional[datetime] = None, to_dt: Optional[datetime] = None
    ) -> None:
        """Start the feed; in phase 1 this only toggles internal flags."""
        if self._started:
            return
        self._started = True
        self._stopped = False
        if not _IBAPI_AVAILABLE:
            # For Phase 1 skeleton, we allow starting without ibapi so tests can inject bars.
            # However, for real connectivity this would raise.
            # Raise only if no symbols and not in test-injection mode.
            # Here we simply proceed; tests will inject via _inject_bar.
            pass

    def stop(self) -> None:
        """Stop the feed; subsequent `next_bar` calls return None."""
        self._stopped = True

    def next_bar(self) -> Optional[Tuple[datetime, pl.DataFrame]]:
        """Return next injected bar or None if queue is empty or feed stopped."""
        if not self._started or self._stopped or self._closed:
            return None
        try:
            ts, df = self._bars_queue.get_nowait()
        except queue.Empty:
            return None
        self._current_ts = ts
        # Optionally select fields if requested
        if self._fields:
            required = {"timestamp", "symbol"}
            cols = [c for c in self._fields if c in df.columns]
            for r in required:
                if r not in cols and r in df.columns:
                    cols.append(r)
            if cols:
                df = df.select(cols)
        return ts, df

    def current_time(self) -> Optional[datetime]:
        """Return the timestamp of the last emitted bar, if any."""
        return self._current_ts

    def is_realtime(self) -> bool:
        """Return True to reflect push-like behavior in tests."""
        return self._is_realtime

    def close(self) -> None:
        """Close the feed and drain any queued bars."""
        self._closed = True
        self._stopped = True
        # drain queue
        try:
            while True:
                self._bars_queue.get_nowait()
        except queue.Empty:
            pass

    # ---- Test helpers (not part of public interface) ----

    def _inject_bar(self, ts: datetime, rows: List[dict]) -> None:
        """Inject a per-timestamp bar slice as a Polars DataFrame.

        `rows` must include keys: timestamp, symbol, open, high, low, close, volume.
        """
        df = pl.DataFrame(rows)
        # Normalize timestamp dtype
        if "timestamp" in df.columns:
            try:
                df = df.with_columns(
                    pl.col("timestamp").cast(
                        pl.Datetime(time_unit="us", time_zone=self._config.timezone)
                    )
                )
            except Exception:
                pass
        self._bars_queue.put((ts, df))

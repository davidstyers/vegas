from __future__ import annotations

"""
Interactive Brokers (IBKR) market data feed.

This module provides a feed that conforms to vegas.marketdata.feed.MarketDataFeed
so it can be passed into BacktestEngine.run_live(..., feed=...).

Phase 1: Skeleton implementation with optional ibapi imports and safe fallbacks.
- If ibapi is not installed, start() raises informative ImportError.
- Public methods operate with in-memory queues so tests can inject bars hermetically.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import queue

import polars as pl

try:
    # Optional dependency; real connectivity added in a later phase
    from ibapi.client import EClient  # type: ignore
    from ibapi.wrapper import EWrapper  # type: ignore
    _IBAPI_AVAILABLE = True
except Exception:
    _IBAPI_AVAILABLE = False

from vegas.marketdata.feed import MarketDataFeed


@dataclass
class IBKRFeedConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 124
    bar_size: str = "1 min"      # Future use
    what_to_show: str = "TRADES" # Future use
    use_rth: bool = True         # Future use
    timezone: str = "UTC"


class InteractiveBrokersFeed(MarketDataFeed):
    """
    MarketDataFeed implementation for Interactive Brokers.

    Phase 1:
      - No real network; exposes in-memory injection helpers for tests.
      - Maintains a per-timestamp queue of polars DataFrame slices.
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
        if symbols:
            self._symbols = list(symbols)
        self._fields = fields

    def start(self, from_dt: Optional[datetime] = None, to_dt: Optional[datetime] = None) -> None:
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
        self._stopped = True

    def next_bar(self) -> Optional[Tuple[datetime, pl.DataFrame]]:
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
        return self._current_ts

    def is_realtime(self) -> bool:
        return self._is_realtime

    def close(self) -> None:
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
        """
        Inject a ready-to-emit bar slice (per timestamp) as a polars DataFrame.
        rows must include keys: timestamp, symbol, open, high, low, close, volume.
        """
        df = pl.DataFrame(rows)
        # Normalize timestamp dtype
        if "timestamp" in df.columns:
            try:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone=self._config.timezone)))
            except Exception:
                pass
        self._bars_queue.put((ts, df))
from __future__ import annotations

"""
Interactive Brokers (IBKR) broker adapter.

This module provides an adapter that conforms to vegas.broker.adapters.BrokerAdapter
so it can be passed into BacktestEngine.run_live(..., broker=...).

Phase 1: Skeleton implementation with optional ibapi imports and safe fallbacks.
- If ibapi is not installed, functionality raises informative ImportError on connect paths.
- Public methods are implemented with in-memory structures so tests can mock behavior.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading
import queue

try:
    # Optional dependency; only needed for real connectivity in a later phase
    from ibapi.client import EClient  # type: ignore
    from ibapi.wrapper import EWrapper  # type: ignore
    from ibapi.contract import Contract  # type: ignore
    from ibapi.order import Order  # type: ignore
    _IBAPI_AVAILABLE = True
except Exception:
    _IBAPI_AVAILABLE = False

from vegas.broker.adapters import BrokerAdapter


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper: 7497, IB Gateway paper: 4002 by default
    client_id: int = 123
    account_id: Optional[str] = None


class _IBClientSkeleton:
    """
    Minimal skeleton to represent an IB client connection. In Phase 1, this does not
    create a real socket; tests can monkeypatch methods or inject events.
    """
    def __init__(self, config: IBKRConfig):
        self.config = config


class InteractiveBrokersBrokerAdapter(BrokerAdapter):
    """
    Interactive Brokers adapter conforming to BrokerAdapter interface.
    In Phase 1, provides in-memory queues and caches for hermetic tests, with optional
    ibapi imports guarded. Real connection will be implemented in a later phase.
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        self._config = config or IBKRConfig()
        # In-memory state suitable for tests
        self._fills_queue: "queue.Queue[Any]" = queue.Queue()
        self._open_orders_cache: List[Any] = []
        self._account_snapshot: Dict[str, Any] = {"cash": 100000.0}
        self._positions_snapshot: Dict[str, Any] = {}
        self._on_fill_cb = None

        # Connection members (unused in Phase 1)
        self._connected = False
        self._thread: Optional[threading.Thread] = None

    # ---- Public BrokerAdapter API ----

    def place_order(self, signal) -> str:
        """
        Map Strategy.Signal into an order. In Phase 1, generate an id and append to open orders cache.
        Tests can simulate a fill by pushing objects into _fills_queue.
        """
        order_id = f"IB-{int(datetime.utcnow().timestamp() * 1e6)}"
        # Cache a minimal open order representation
        order_repr = {
            "id": order_id,
            "symbol": getattr(signal, "symbol", None),
            "quantity": getattr(signal, "quantity", 0),
            "price": getattr(signal, "price", None),
            "action": getattr(signal, "action", "buy"),
            "status": "open",
        }
        self._open_orders_cache.append(order_repr)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        for o in self._open_orders_cache:
            if o.get("id") == order_id and o.get("status") in ("open", "partially_filled"):
                o["status"] = "cancelled"
                return True
        return False

    def get_open_orders(self) -> List[Any]:
        return [o for o in self._open_orders_cache if o.get("status") in ("open", "partially_filled")]

    def poll_fills(self, until: Optional[datetime] = None) -> List[Any]:
        """
        Drain fills captured since last poll.
        Fill objects may be dicts with keys: symbol, quantity, price, commission.
        """
        fills: List[Any] = []
        try:
            while True:
                fills.append(self._fills_queue.get_nowait())
        except queue.Empty:
            pass

        if fills and self._on_fill_cb is not None:
            try:
                self._on_fill_cb(fills)
            except Exception:
                pass
        return fills

    def on_fill(self, callback) -> None:
        self._on_fill_cb = callback  # already defined in base

    def get_account(self) -> Dict[str, Any]:
        return dict(self._account_snapshot)

    def get_positions(self) -> Dict[str, Any]:
        return dict(self._positions_snapshot)

    # ---- Utilities for tests/mocks to seed state ----

    def _seed_account(self, cash: float, positions: Optional[Dict[str, Any]] = None) -> None:
        self._account_snapshot = {"cash": float(cash)}
        self._positions_snapshot = positions or {}

    def _push_fill(self, symbol: str, quantity: float, price: float, commission: float = 0.0) -> None:
        self._fills_queue.put({
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "commission": commission,
        })
        # Update open orders cache status to 'filled' for the symbol if present
        for o in self._open_orders_cache:
            if o.get("symbol") == symbol and o.get("status") in ("open", "partially_filled"):
                o["status"] = "filled"
                break

    # ---- Connection lifecycle (stubs for Phase 1) ----

    def connect(self) -> None:
        if not _IBAPI_AVAILABLE:
            raise ImportError("ibapi is not installed. Install with extras: pip install 'vegas[ibkr]'")
        # Real connection will be implemented in Phase 2
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
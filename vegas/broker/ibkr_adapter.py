from __future__ import annotations

"""
Interactive Brokers (IBKR) broker adapter.

This module provides an adapter that conforms to `vegas.broker.adapters.BrokerAdapter`
so it can be passed into `BacktestEngine.run_live(..., broker=...)`.

Production-ready implementation characteristics:
- Optional dependency on `ibapi` with graceful fallbacks for unit testing without TWS/Gateway
- Threaded, non-blocking event loop for the IB API client
- Synchronous wrappers around request/response style IB API endpoints using Events
- Structured logging and selective handling of common error codes
- Environment-driven configuration (host/port/client_id/account_id)
- Basic market data subscription management for L1 (bid/ask/last)

Notes on interface alignment:
- The Vegas engine currently defines `BrokerAdapter` rather than a `LiveBroker` interface.
  This class fully implements the `BrokerAdapter` API and adds connection and market-data
  methods to satisfy live-trading needs. An alias `InteractiveBrokersBroker` is provided
  for semantic clarity.
"""

import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    # Optional dependency; only needed for real connectivity
    from ibapi.client import EClient  # type: ignore
    from ibapi.common import TickerId  # type: ignore
    from ibapi.contract import Contract  # type: ignore
    from ibapi.order import Order  # type: ignore
    from ibapi.wrapper import EWrapper  # type: ignore

    _IBAPI_AVAILABLE = True
except Exception:
    _IBAPI_AVAILABLE = False

from vegas.broker.adapters import BrokerAdapter


@dataclass
class IBKRConfig:
    """Configuration for `InteractiveBrokersBrokerAdapter` connection params.

    Values default to typical TWS paper trading settings. All parameters can be
    supplied via environment variables:
      - IB_HOST (default: 127.0.0.1)
      - IB_PORT (default: 7497)
      - IB_CLIENT_ID (default: 123)
      - IB_ACCOUNT_ID (optional)
    """

    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper: 7497, IB Gateway paper: 4002 by default
    client_id: int = 123
    account_id: Optional[str] = None
    auto_reconnect: bool = True
    reconnect_min_delay: float = 2.0
    reconnect_max_delay: float = 30.0

    @staticmethod
    def from_env() -> "IBKRConfig":
        host = os.getenv("IB_HOST", "127.0.0.1")
        port = int(os.getenv("IB_PORT", "7497"))
        client_id = int(os.getenv("IB_CLIENT_ID", "123"))
        account_id = os.getenv("IB_ACCOUNT_ID")
        auto_reconnect = os.getenv("IB_AUTO_RECONNECT", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        min_delay = float(os.getenv("IB_RECONNECT_MIN_DELAY", "2.0"))
        max_delay = float(os.getenv("IB_RECONNECT_MAX_DELAY", "30.0"))
        return IBKRConfig(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id,
            auto_reconnect=auto_reconnect,
            reconnect_min_delay=min_delay,
            reconnect_max_delay=max_delay,
        )


class _IBClientSkeleton:
    """Minimal stand-in for an IB client connection (no real sockets)."""

    def __init__(self, config: IBKRConfig):
        self.config = config


class _IBResponseBook:
    """Thread-safe cache for IB API responses with request-scoped Events.

    This object stores transient results for request/response style endpoints
    like account summary, open orders, and positions. Methods provide
    synchronization primitives (Events) for callers to wait until completion.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._account_summary: Dict[int, Dict[str, Any]] = {}
        self._account_summary_evt: Dict[int, threading.Event] = {}

        self._open_orders: Dict[int, List[Dict[str, Any]]] = {}
        self._open_orders_evt: Dict[int, threading.Event] = {}

        self._positions: Dict[int, List[Dict[str, Any]]] = {}
        self._positions_evt: Dict[int, threading.Event] = {}

        self._next_order_id: Optional[int] = None
        self._next_order_id_evt = threading.Event()

        self._quotes_by_ticker: Dict[int, Dict[str, Any]] = {}
        self._ticker_to_symbol: Dict[int, str] = {}
        self._symbol_to_ticker: Dict[str, int] = {}

    # ---- next valid id ----
    def set_next_order_id(self, order_id: int) -> None:
        with self._lock:
            self._next_order_id = order_id
            self._next_order_id_evt.set()

    def wait_for_next_order_id(self, timeout: float = 5.0) -> Optional[int]:
        self._next_order_id_evt.wait(timeout=timeout)
        with self._lock:
            return self._next_order_id

    def consume_and_increment_order_id(self) -> Optional[int]:
        with self._lock:
            if self._next_order_id is None:
                return None
            oid = self._next_order_id
            self._next_order_id += 1
            return oid

    # ---- account summary ----
    def begin_account_summary(self, req_id: int) -> None:
        with self._lock:
            self._account_summary[req_id] = {}
            self._account_summary_evt[req_id] = threading.Event()

    def on_account_summary(
        self, req_id: int, tag: str, value: str, currency: str
    ) -> None:
        with self._lock:
            if req_id not in self._account_summary:
                self._account_summary[req_id] = {}
            # Most tags are numeric strings; best-effort cast
            try:
                # Keep currency-specific keys when provided
                val: Any = float(value)
            except Exception:
                val = value
            self._account_summary[req_id][tag] = val
            if currency:
                self._account_summary[req_id][f"{tag}.currency"] = currency

    def end_account_summary(self, req_id: int) -> None:
        with self._lock:
            evt = self._account_summary_evt.get(req_id)
            if evt:
                evt.set()

    def await_account_summary(
        self, req_id: int, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        evt = self._account_summary_evt.get(req_id)
        if evt is None:
            return None
        evt.wait(timeout=timeout)
        with self._lock:
            return dict(self._account_summary.get(req_id, {}))

    # ---- open orders ----
    def begin_open_orders(self, req_id: int) -> None:
        with self._lock:
            self._open_orders[req_id] = []
            self._open_orders_evt[req_id] = threading.Event()

    def on_open_order(self, req_id: int, order_data: Dict[str, Any]) -> None:
        with self._lock:
            self._open_orders.setdefault(req_id, []).append(order_data)

    def end_open_orders(self, req_id: int) -> None:
        with self._lock:
            evt = self._open_orders_evt.get(req_id)
            if evt:
                evt.set()

    def await_open_orders(
        self, req_id: int, timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        evt = self._open_orders_evt.get(req_id)
        if evt is None:
            return []
        evt.wait(timeout=timeout)
        with self._lock:
            return list(self._open_orders.get(req_id, []))

    # ---- positions ----
    def begin_positions(self, req_id: int) -> None:
        with self._lock:
            self._positions[req_id] = []
            self._positions_evt[req_id] = threading.Event()

    def on_position(self, req_id: int, pos_row: Dict[str, Any]) -> None:
        with self._lock:
            self._positions.setdefault(req_id, []).append(pos_row)

    def end_positions(self, req_id: int) -> None:
        with self._lock:
            evt = self._positions_evt.get(req_id)
            if evt:
                evt.set()

    def await_positions(
        self, req_id: int, timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        evt = self._positions_evt.get(req_id)
        if evt is None:
            return []
        evt.wait(timeout=timeout)
        with self._lock:
            return list(self._positions.get(req_id, []))

    # ---- quotes ----
    def register_ticker(self, ticker_id: int, symbol: str) -> None:
        with self._lock:
            self._ticker_to_symbol[ticker_id] = symbol
            self._symbol_to_ticker[symbol] = ticker_id
            self._quotes_by_ticker.setdefault(ticker_id, {})

    def unregister_ticker(self, ticker_id: int) -> None:
        with self._lock:
            self._quotes_by_ticker.pop(ticker_id, None)
            sym = self._ticker_to_symbol.pop(ticker_id, None)
            if sym:
                self._symbol_to_ticker.pop(sym, None)

    def on_tick(
        self, ticker_id: int, field_name: str, value: float
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            if ticker_id not in self._quotes_by_ticker:
                self._quotes_by_ticker[ticker_id] = {}
            self._quotes_by_ticker[ticker_id][field_name] = value
            symbol = self._ticker_to_symbol.get(ticker_id)
            snap = dict(self._quotes_by_ticker[ticker_id])
            return (symbol, snap) if symbol else None

    def get_quote_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            tid = self._symbol_to_ticker.get(symbol)
            if tid is None:
                return None
            return dict(self._quotes_by_ticker.get(tid, {}))


if _IBAPI_AVAILABLE:

    class _IBApiApp(EWrapper, EClient):
        """Thin wrapper that forwards EWrapper callbacks into `_IBResponseBook` and queues.

        The broker instance provides the response book and the fills queue.
        """

        def __init__(
            self,
            response_book: _IBResponseBook,
            fills_queue: "queue.Queue[Any]",
            logger: logging.Logger,
            on_connection_state: Optional[Callable[[str], None]] = None,
            on_quote: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        ):
            EClient.__init__(self, self)
            self._resp = response_book
            self._fills_queue = fills_queue
            self._logger = logger
            self._on_connection_state = on_connection_state
            self._on_quote = on_quote

        # ---- Lifecycle ----
        def nextValidId(self, orderId: int):  # noqa: N802 (IBAPI camelCase)
            self._logger.debug(f"IB nextValidId: {orderId}")
            self._resp.set_next_order_id(orderId)

        def error(
            self,
            reqId: int,
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ):
            # Common connection codes: 1100 (down), 1101/1102 (up), 502 (connect failed)
            level = logging.WARNING if errorCode >= 1000 else logging.ERROR
            self._logger.log(
                level, f"IB error: reqId={reqId}, code={errorCode}, msg={errorString}"
            )
            if errorCode == 1100 and self._on_connection_state:
                self._on_connection_state("down")
            elif errorCode in (1101, 1102) and self._on_connection_state:
                self._on_connection_state("up")

        def connectionClosed(self):
            self._logger.warning("IB connectionClosed callback received")
            if self._on_connection_state:
                self._on_connection_state("down")

        # ---- Account Summary ----
        def accountSummary(
            self, reqId: int, account: str, tag: str, value: str, currency: str
        ):
            self._resp.on_account_summary(reqId, tag, value, currency)

        def accountSummaryEnd(self, reqId: int):
            self._resp.end_account_summary(reqId)

        # ---- Positions ----
        def position(self, account: str, contract, position: float, avgCost: float):
            # position() has no reqId, so we key under a single active req
            # We'll support only one outstanding positions request at a time.
            self._resp.on_position(
                0,
                {
                    "account": account,
                    "symbol": getattr(contract, "symbol", None),
                    "secType": getattr(contract, "secType", None),
                    "position": position,
                    "avgCost": avgCost,
                },
            )

        def positionEnd(self):
            self._resp.end_positions(0)

        # ---- Open Orders ----
        def openOrder(self, orderId: int, contract, order: Order, orderState):
            # openOrder() has no reqId; key under a single active request
            self._resp.on_open_order(
                0,
                {
                    "orderId": orderId,
                    "symbol": getattr(contract, "symbol", None),
                    "action": getattr(order, "action", None),
                    "totalQuantity": getattr(order, "totalQuantity", None),
                    "orderType": getattr(order, "orderType", None),
                    "lmtPrice": getattr(order, "lmtPrice", None),
                    "auxPrice": getattr(order, "auxPrice", None),
                    "status": getattr(orderState, "status", None),
                },
            )

        def openOrderEnd(self):
            self._resp.end_open_orders(0)

        def orderStatus(
            self,
            orderId: int,
            status: str,
            filled: float,
            remaining: float,
            avgFillPrice: float,
            permId: int,
            parentId: int,
            lastFillPrice: float,
            clientId: int,
            whyHeld: str,
            mktCapPrice: float,
        ):
            # Push fills when filled quantity increases
            if lastFillPrice and filled:
                self._fills_queue.put(
                    {
                        "order_id": orderId,
                        "price": lastFillPrice,
                        "filled": filled,
                        "remaining": remaining,
                        "avg_price": avgFillPrice,
                        "timestamp": datetime.utcnow(),
                    }
                )

        # ---- Executions / Commission (optional richer details) ----
        def execDetails(self, reqId: int, contract, execution):
            self._fills_queue.put(
                {
                    "req_id": reqId,
                    "symbol": getattr(contract, "symbol", None),
                    "price": getattr(execution, "price", None),
                    "qty": getattr(execution, "shares", None),
                    "time": getattr(execution, "time", None),
                    "order_id": getattr(execution, "orderId", None),
                }
            )

        # ---- Market Data (L1) ----
        # tickPrice/tickSize fields per https://interactivebrokers.github.io/tws-api/tick_types.html
        def tickPrice(self, tickerId: TickerId, field: int, price: float, attrib):
            mapping = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
            name = mapping.get(field)
            if name:
                out = self._resp.on_tick(int(tickerId), name, float(price))
                if out:
                    symbol, snap = out
                    self._logger.debug(f"tickPrice {symbol}: {name}={price}")
                    if self._on_quote and symbol:
                        try:
                            self._on_quote(symbol, snap)
                        except Exception:
                            pass

        def tickSize(self, tickerId: TickerId, field: int, size: int):
            mapping = {0: "bid_size", 3: "ask_size", 5: "last_size", 8: "volume"}
            name = mapping.get(field)
            if name:
                out = self._resp.on_tick(int(tickerId), name, float(size))
                if out:
                    symbol, snap = out
                    self._logger.debug(f"tickSize {symbol}: {name}={size}")
                    if self._on_quote and symbol:
                        try:
                            self._on_quote(symbol, snap)
                        except Exception:
                            pass


class InteractiveBrokersBrokerAdapter(BrokerAdapter):
    """Interactive Brokers adapter conforming to `BrokerAdapter`.

    The class supports two operational modes:
    - Offline mode (no `ibapi` or not connected): methods operate against in-memory
      caches to enable unit testing without network dependencies.
    - Online mode (connected to TWS/Gateway): methods delegate to `ibapi` and
      synchronize responses via an internal response book.
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        self._logger = logging.getLogger("vegas.ibkr")
        self._config = config or IBKRConfig.from_env()

        # In-memory state (always present; used in offline mode and as caches online)
        self._fills_queue: "queue.Queue[Any]" = queue.Queue()
        self._open_orders_cache: List[Dict[str, Any]] = []
        self._account_snapshot: Dict[str, Any] = {
            "cash": 0.0,
            "equity": 0.0,
            "buying_power": 0.0,
        }
        self._positions_snapshot: Dict[str, Any] = {}
        self._on_fill_cb: Optional[Callable[[List[Any]], None]] = None

        # Connection members
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._resp_book = _IBResponseBook()
        self._app = _IBClientSkeleton(self._config) if not _IBAPI_AVAILABLE else None
        # Reconnection support
        self._reconnect_thread: Optional[threading.Thread] = None
        self._reconnect_stop = threading.Event()

        # Market data subscription bookkeeping
        self._next_ticker_id = 1000
        self._on_quote_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None

    # ---- Public BrokerAdapter API ----

    def place_order(self, signal) -> str:
        """Place an order based on a `Signal`.

        - Online mode: build Contract/Order and submit via IB API.
        - Offline mode: append to in-memory open orders cache.
        """
        symbol = getattr(signal, "symbol", None)
        qty = float(getattr(signal, "quantity", 0.0) or 0.0)
        action = "BUY" if qty > 0 else "SELL"
        order_type, fields = self._resolve_order_fields(signal)
        abs_qty = abs(qty)

        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            contract = self._build_stock_contract(symbol)
            order = Order()
            order.action = action
            order.totalQuantity = abs_qty
            order.orderType = order_type
            # Optional fields
            if "lmtPrice" in fields:
                order.lmtPrice = fields["lmtPrice"]
            if "auxPrice" in fields:
                order.auxPrice = fields["auxPrice"]
            order.tif = "DAY"

            order_id = self._resp_book.consume_and_increment_order_id()
            if order_id is None:
                # Request an id and wait briefly
                self._app.reqIds(-1)
                order_id = self._resp_book.wait_for_next_order_id(timeout=5.0)
                if order_id is None:
                    raise RuntimeError("Failed to obtain next order id from IB API")
                order_id = self._resp_book.consume_and_increment_order_id()

            assert order_id is not None
            self._app.placeOrder(order_id, contract, order)
            self._logger.info(
                f"Submitted order {order_id} {action} {abs_qty} {symbol} type={order_type} {fields}"
            )
            # Cache a minimal open order representation
            self._open_orders_cache.append(
                {
                    "id": str(order_id),
                    "symbol": symbol,
                    "quantity": qty,
                    "action": action,
                    "type": order_type,
                    **fields,
                    "status": "open",
                }
            )
            return str(order_id)

        # Offline fallback
        order_id = f"IB-{int(datetime.utcnow().timestamp() * 1e6)}"
        self._open_orders_cache.append(
            {
                "id": order_id,
                "symbol": symbol,
                "quantity": qty,
                "action": action,
                "type": order_type,
                **fields,
                "status": "open",
            }
        )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            try:
                self._app.cancelOrder(int(order_id))
            except Exception:
                # allow non-int ids from offline mode
                pass
        for o in self._open_orders_cache:
            if str(o.get("id")) == str(order_id) and o.get("status") in (
                "open",
                "partially_filled",
            ):
                o["status"] = "cancelled"
                return True
        return False

    def get_open_orders(self) -> List[Any]:
        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            # Request current open orders synchronously
            self._resp_book.begin_open_orders(0)
            self._app.reqOpenOrders()
            orders = self._resp_book.await_open_orders(0, timeout=5.0)
            # Reconcile cache lightly
            return orders
        return [
            o
            for o in self._open_orders_cache
            if o.get("status") in ("open", "partially_filled")
        ]

    def poll_fills(self, until: Optional[datetime] = None) -> List[Any]:
        """Drain fills captured since last poll (dicts include at least price/qty/order_id).

        If a fill callback was registered via `on_fill`, it will be invoked with
        the drained fills list.
        """
        fills: List[Any] = []
        try:
            while True:
                item = self._fills_queue.get_nowait()
                fills.append(item)
        except queue.Empty:
            pass

        if fills and self._on_fill_cb is not None:
            try:
                self._on_fill_cb(fills)
            except Exception:
                pass
        return fills

    def on_fill(self, callback) -> None:
        self._on_fill_cb = callback

    def get_account(self) -> Dict[str, Any]:
        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            req_id = 9001
            self._resp_book.begin_account_summary(req_id)
            # A few common tags for balances and buying power
            tags = "TotalCashValue,NetLiquidation,AvailableFunds,BuyingPower,Currency"
            self._app.reqAccountSummary(req_id, "All", tags)
            summary = self._resp_book.await_account_summary(req_id, timeout=5.0) or {}
            try:
                self._app.cancelAccountSummary(req_id)
            except Exception:
                pass
            # Minimal normalization
            out = {
                "cash": summary.get("TotalCashValue", 0.0),
                "equity": summary.get("NetLiquidation", 0.0),
                "available_funds": summary.get("AvailableFunds", 0.0),
                "buying_power": summary.get("BuyingPower", 0.0),
                "currency": summary.get(
                    "Currency", summary.get("TotalCashValue.currency", "USD")
                ),
            }
            self._account_snapshot.update(out)
            return dict(self._account_snapshot)
        return dict(self._account_snapshot)

    # Optional alias used by some callers/specs
    def get_account_summary(self) -> Dict[str, Any]:
        return self.get_account()

    def get_positions(self) -> Dict[str, Any]:
        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            # IB positions callbacks do not include reqId, we base on a single slot (0)
            self._resp_book.begin_positions(0)
            self._app.reqPositions()
            rows = self._resp_book.await_positions(0, timeout=5.0)
            try:
                self._app.cancelPositions()
            except Exception:
                pass
            pos: Dict[str, Any] = {}
            for r in rows:
                sym = r.get("symbol")
                if not sym:
                    continue
                pos[sym] = {
                    "quantity": r.get("position", 0.0),
                    "avg_cost": r.get("avgCost", 0.0),
                }
            self._positions_snapshot = pos
            return dict(self._positions_snapshot)
        return dict(self._positions_snapshot)

    def _handle_connection_state(self, state: str) -> None:
        if state == "down":
            self._logger.warning("IB connection reported DOWN; marking disconnected")
            self._connected = False
        elif state == "up":
            self._logger.info("IB connection reported UP")

    def _handle_quote_update(self, symbol: str, snapshot: Dict[str, Any]) -> None:
        if self._on_quote_cb is not None:
            try:
                self._on_quote_cb(symbol, snapshot)
            except Exception:
                pass

    def _reconnect_watchdog(self) -> None:
        if not self._config.auto_reconnect:
            return
        backoff = self._config.reconnect_min_delay
        while not self._reconnect_stop.is_set():
            if not self._connected:
                self._logger.info(
                    f"Reconnect watchdog attempting to connect in {backoff:.1f}s ..."
                )
                self._reconnect_stop.wait(backoff)
                if self._reconnect_stop.is_set():
                    break
                try:
                    self.connect()
                    backoff = self._config.reconnect_min_delay
                except Exception as e:
                    self._logger.error(f"Reconnect attempt failed: {e}")
                    backoff = min(
                        self._config.reconnect_max_delay,
                        max(self._config.reconnect_min_delay, backoff * 1.5),
                    )
            else:
                # Sleep lightly; check state again
                self._reconnect_stop.wait(2.0)

    # ---- Utilities for tests/mocks to seed state ----

    def _seed_account(
        self, cash: float, positions: Optional[Dict[str, Any]] = None
    ) -> None:
        self._account_snapshot = {
            "cash": float(cash),
            "equity": float(cash),
            "buying_power": float(cash),
        }
        self._positions_snapshot = positions or {}

    def _push_fill(
        self, symbol: str, quantity: float, price: float, commission: float = 0.0
    ) -> None:
        self._fills_queue.put(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "commission": commission,
            }
        )
        # Update open orders cache status to 'filled' for the symbol if present
        for o in self._open_orders_cache:
            if o.get("symbol") == symbol and o.get("status") in (
                "open",
                "partially_filled",
            ):
                o["status"] = "filled"
                break

    # ---- Connection lifecycle ----

    def connect(self) -> None:
        if not _IBAPI_AVAILABLE:
            raise ImportError(
                "ibapi is not installed. Install with extras: pip install 'vegas[ibkr]'"
            )

        if self._connected and isinstance(self._app, _IBApiApp):
            return

        self._resp_book = _IBResponseBook()
        self._app = _IBApiApp(
            self._resp_book,
            self._fills_queue,
            self._logger,
            on_connection_state=self._handle_connection_state,
            on_quote=self._handle_quote_update,
        )
        # Establish TCP/SSL connection
        self._logger.info(
            f"Connecting to IBKR at {self._config.host}:{self._config.port} as client {self._config.client_id}"
        )
        try:
            ok = self._app.connect(
                self._config.host, self._config.port, clientId=self._config.client_id
            )
        except Exception as e:
            self._logger.error(f"Failed to initiate connection to IBKR: {e}")
            raise

        # Spin the network thread
        self._thread = threading.Thread(
            target=self._app.run, name="IBApiThread", daemon=True
        )
        self._thread.start()

        # Wait for nextValidId
        oid = self._resp_book.wait_for_next_order_id(timeout=5.0)
        if oid is None:
            self._logger.warning(
                "Did not receive nextValidId within timeout. Continuing anyway."
            )
        else:
            self._logger.info(f"Connected. Next order id: {oid}")
        self._connected = True
        # Start auto-reconnect watchdog if enabled
        if self._config.auto_reconnect and (
            self._reconnect_thread is None or not self._reconnect_thread.is_alive()
        ):
            self._reconnect_stop.clear()
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_watchdog, name="IBReconnect", daemon=True
            )
            self._reconnect_thread.start()

    def disconnect(self) -> None:
        if _IBAPI_AVAILABLE and isinstance(self._app, _IBApiApp):
            try:
                self._app.disconnect()
            except Exception:
                pass
        # Stop reconnect thread
        self._reconnect_stop.set()
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            try:
                self._reconnect_thread.join(timeout=2.0)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            # Give the thread a moment to unwind
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass
        self._connected = False
        self._app = None

    # ---- Market Data Subscriptions (L1) ----

    def subscribe_market_data(self, symbol: str) -> Optional[int]:
        if not (
            _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp)
        ):
            # Offline mode: simulate allocation of a ticker id and store mapping
            tid = self._next_ticker_id
            self._next_ticker_id += 1
            self._resp_book.register_ticker(tid, symbol)
            return tid

        contract = self._build_stock_contract(symbol)
        tid = self._next_ticker_id
        self._next_ticker_id += 1
        self._resp_book.register_ticker(tid, symbol)
        self._app.reqMktData(tid, contract, "", False, False, [])
        self._logger.info(f"Subscribed L1 for {symbol} (tickerId={tid})")
        return tid

    def unsubscribe_market_data(self, symbol: str) -> None:
        tid = self._resp_book._symbol_to_ticker.get(symbol)
        if tid is None:
            return
        if _IBAPI_AVAILABLE and self._connected and isinstance(self._app, _IBApiApp):
            try:
                self._app.cancelMktData(tid)
            except Exception:
                pass
        self._resp_book.unregister_ticker(tid)
        self._logger.info(f"Unsubscribed L1 for {symbol} (tickerId={tid})")

    def on_quote(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback for L1 quote updates per symbol.

        Note: This is optional and not part of `BrokerAdapter`, but handy for live integrations.
        """
        self._on_quote_cb = callback

    # ---- Helpers ----

    @staticmethod
    def _build_stock_contract(symbol: str) -> "Contract":
        if not _IBAPI_AVAILABLE:
            raise ImportError("ibapi is not installed")
        c = Contract()
        c.symbol = symbol
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    @staticmethod
    def _resolve_order_fields(signal) -> Tuple[str, Dict[str, Any]]:
        """Resolve IB order type and fields from a strategy `Signal`.

        Supported types: Market, Limit, Stop, Stop Limit.
        Returns a tuple of (orderType, fieldsDict).
        """
        # Explicit type string, normalize
        explicit_type = getattr(signal, "order_type", None)
        limit_price = getattr(signal, "limit_price", None) or getattr(
            signal, "price", None
        )
        stop_price = getattr(signal, "stop_price", None)

        if explicit_type:
            et = str(explicit_type).strip().lower()
        else:
            et = None

        if et in {"market", "mkt"} or (
            et is None and limit_price is None and stop_price is None
        ):
            return "MKT", {}
        if et in {"limit", "lmt"} or (limit_price is not None and stop_price is None):
            return "LMT", {"lmtPrice": float(limit_price)}
        if et in {"stop", "stp"} or (stop_price is not None and limit_price is None):
            return "STP", {"auxPrice": float(stop_price)}
        if et in {"stop_limit", "stp lmt", "stp_lmt", "stop limit"} or (
            stop_price is not None and limit_price is not None
        ):
            return "STP LMT", {
                "auxPrice": float(stop_price),
                "lmtPrice": float(limit_price),
            }

        # Fallback to market
        return "MKT", {}


# Semantic alias requested in requirements
InteractiveBrokersBroker = InteractiveBrokersBrokerAdapter

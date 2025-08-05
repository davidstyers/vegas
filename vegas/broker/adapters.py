from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl


class BrokerAdapter:
    """
    Abstract broker adapter interface.
    """

    def place_order(self, signal) -> str:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_open_orders(self) -> List[Any]:
        raise NotImplementedError

    def poll_fills(self, until: Optional[datetime] = None) -> List[Any]:
        """
        Return a list of fill/transaction-like records since the last poll.
        """
        raise NotImplementedError

    def on_fill(self, callback) -> None:
        """
        Optional: register a callback invoked when fills are detected.
        """
        self._on_fill_cb = callback  # type: ignore[attr-defined]

    def get_account(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_positions(self) -> Dict[str, Any]:
        raise NotImplementedError


class SimulatedBrokerAdapter(BrokerAdapter):
    """
    Adapter that wraps the existing Broker to provide backtest-style execution.
    """

    def __init__(self, broker):
        self._broker = broker
        self._pending_fills: List[Any] = []
        self._on_fill_cb = None

    def place_order(self, signal) -> str:
        """
        Delegate to broker.place_order and return order id.
        """
        order = self._broker.place_order(signal)
        return getattr(order, "id", str(order))

    def cancel_order(self, order_id: str) -> bool:
        return self._broker.cancel_order(order_id)

    def get_open_orders(self) -> List[Any]:
        # Expose the broker's open orders list if available
        try:
            return [o for o in getattr(self._broker, "orders", []) if getattr(o, "status", "") in ("open", "partially_filled")]
        except Exception:
            return []

    def simulate_execute(self, market_data: Dict[str, "pl.DataFrame"], timestamp: datetime) -> List[Any]:
        """
        Execute pending orders against the provided market snapshot using
        the underlying Broker.execute_orders and store fills locally.
        """
        # The underlying Broker.execute_orders expects pandas DataFrames per symbol.
        # The engine will control conversion to its internal polars format when updating portfolio.
        pd_market: Dict[str, Any] = {}
        for sym, df in market_data.items():
            # df is polars.DataFrame; convert to pandas lazily for broker compat
            pd_market[sym] = df.to_pandas()

        transactions = self._broker.execute_orders(pd_market, timestamp)  # returns list of Transaction dataclasses
        if transactions:
            self._pending_fills.extend(transactions)
            if self._on_fill_cb is not None:
                try:
                    self._on_fill_cb(transactions)
                except Exception:
                    pass
        return transactions

    def poll_fills(self, until: Optional[datetime] = None) -> List[Any]:
        """
        Return fills captured since last poll. 'until' is accepted for parity but not used here.
        """
        fills = list(self._pending_fills)
        self._pending_fills.clear()
        return fills

    def get_account(self) -> Dict[str, Any]:
        return self._broker.get_account()

    def get_positions(self) -> Dict[str, Any]:
        # Expose a simplified positions mapping
        try:
            return {
                sym: {
                    "quantity": pos.quantity,
                    "cost_basis": pos.cost_basis,
                    "market_value": pos.market_value,
                }
                for sym, pos in getattr(self._broker, "positions", {}).items()
            }
        except Exception:
            return {}
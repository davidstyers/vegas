"""Broker simulation layer for the Vegas backtesting engine.

This module provides broker simulation for executing orders with slippage
and commission models, tracking positions, and maintaining transaction history.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import polars as pl
import uuid

from vegas.broker.slippage import SlippageModel, FixedSlippageModel
from vegas.broker.commission import CommissionModel, FixedCommissionModel, PerShareCommissionModel
from vegas.strategy import Signal


class OrderStatus(Enum):
    """Order status enumeration."""
    OPEN = 'open'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAIL_STOP = 'trail_stop'
    TRAIL_STOP_LIMIT = 'trail_stop_limit'


@dataclass
class Order:
    """Order representation with extended types (stop, stop-limit, trailing) and OCO/brackets.
    
    Attributes:
        id: Unique order identifier
        symbol: Trading symbol
        quantity: Order quantity (positive for buy, negative for sell). Sign implies side.
        order_type: MARKET, LIMIT, STOP, STOP_LIMIT, TRAIL_STOP, TRAIL_STOP_LIMIT
        limit_price: Price limit for LIMIT or for post-trigger LIMIT conversion
        stop_price: Static stop trigger price (STOP/STOP_LIMIT)
        trail_amount: Absolute dollar trail distance (mutually exclusive with trail_percent)
        trail_percent: Percent trail (0.30 = 30%) (mutually exclusive with trail_amount)
        trigger_on_range: If True, evaluate triggers against bar high/low range (recommended)
        created_at: Order creation timestamp
        status: Current order status
        filled_quantity: Quantity filled so far
        filled_price: Average fill price for filled quantity
        # Runtime state for trailing:
        trail_ref_price: Highest (long) or lowest (short) reference since activation
        dynamic_stop: Current computed trailing stop level
        triggered: Whether the stop/trigger condition has fired (for STOP/STOP_LIMIT/TRAIL_*)
        # Bracket/OCO:
        parent_id: If set, this order is a child of a parent order (e.g. bracket)
        oco_group_id: If set, this order is part of an OCO group; when one fills, others cancel
        bracket_role: Optional role hint for readability: "take_profit" or "stop_loss"
    """
    id: str
    symbol: str
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    trigger_on_range: bool = True
    created_at: datetime = None
    status: OrderStatus = OrderStatus.OPEN
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    # runtime fields
    trail_ref_price: Optional[float] = None
    dynamic_stop: Optional[float] = None
    triggered: bool = False
    # linking fields
    parent_id: Optional[str] = None
    oco_group_id: Optional[str] = None
    bracket_role: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with default values and validate fields."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.id is None:
            self.id = str(uuid.uuid4())
        # Basic validation
        if self.order_type in (OrderType.TRAIL_STOP, OrderType.TRAIL_STOP_LIMIT):
            if (self.trail_amount is None) == (self.trail_percent is None):
                # Must provide exactly one
                raise ValueError("Provide exactly one of trail_amount or trail_percent for trailing orders.")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("stop_price is required for STOP/STOP_LIMIT orders.")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TRAIL_STOP_LIMIT) and self.limit_price is None:
            # Allow None for pre-trigger trailing stop limit; but STOP_LIMIT requires a limit on trigger
            if self.order_type != OrderType.TRAIL_STOP_LIMIT:
                raise ValueError("limit_price is required for LIMIT/STOP_LIMIT orders.")


@dataclass
class Transaction:
    """Transaction record.
    
    Attributes:
        id: Unique transaction identifier
        order_id: Reference to the originating order
        symbol: Trading symbol
        quantity: Transaction quantity (positive for buy, negative for sell)
        price: Execution price
        commission: Transaction fee
        timestamp: Transaction timestamp
    """
    id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    
    def __post_init__(self):
        """Initialize with default values."""
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def value(self) -> float:
        """Calculate transaction value.
        
        Returns:
            Transaction value (positive for buys, negative for sells)
        """
        return self.quantity * self.price


class Position:
    """Position representation for a single symbol.
    
    Attributes:
        symbol: Trading symbol
        quantity: Current position size
        cost_basis: Average cost basis
        market_value: Current market value
    """
    
    def __init__(self, symbol: str):
        """Initialize a position.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.quantity = 0.0
        self.cost_basis = 0.0
        self.market_value = 0.0
    
    def update(self, quantity: float, price: float) -> None:
        """Update position with a new transaction.
        
        Args:
            quantity: Transaction quantity (positive for buy, negative for sell)
            price: Transaction price
        """
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.cost_basis = price
        elif quantity * self.quantity > 0:
            # Adding to position
            total_cost = (self.cost_basis * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.cost_basis = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing position or flipping
            self.quantity += quantity
            if self.quantity == 0:
                self.cost_basis = 0
            elif self.quantity * (self.quantity - quantity) < 0:
                # Position flipped from long to short or vice versa
                self.cost_basis = price
    
    def update_market_value(self, price: float) -> None:
        """Update position market value.
        
        Args:
            price: Current market price
        """
        self.market_value = self.quantity * price
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss.
        
        Returns:
            Unrealized profit/loss
        """
        return self.market_value - (self.quantity * self.cost_basis)


class Broker:
    """Broker simulation for executing orders and tracking positions.
    
    This class simulates order execution with slippage and commission models,
    tracks positions, and maintains transaction history.
    """
    
    def __init__(self, initial_cash: float = 100000.0,
                slippage_model: Optional[SlippageModel] = None,
                commission_model: Optional[CommissionModel] = None,
                data_portal: Optional[object] = None):
        """Initialize the broker.
        
        Args:
            initial_cash: Initial cash balance
            slippage_model: Slippage model for order execution
            commission_model: Commission model for order execution
        """
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.transactions: List[Transaction] = []
        
        # Default models
        self.slippage_model = slippage_model or FixedSlippageModel()
        # Default to zero-cost per-share if none provided; engine/strategy can override
        self.commission_model: CommissionModel = commission_model or PerShareCommissionModel(cost_per_share=0.0, min_trade_cost=0.0)
        # Optional DataPortal reference
        self._data_portal = data_portal

    def set_data_portal(self, data_portal) -> None:
        """Inject a DataPortal for brokers that need to fetch market data internally."""
        self._data_portal = data_portal
    
    def place_order(self, signal: Signal) -> Order:
        """Place an order based on a signal.
        
        Rules:
          - Side is inferred solely from the sign of quantity (quantity > 0 = buy, quantity < 0 = sell).
          - Accepts order_type from the signal as one of:
              "market", "limit", "stop", "stop_limit", "trail_stop", "trail_stop_limit" (case-insensitive).
          - Backward compatibility:
              * If order_type is missing, derive from provided fields:
                  - If stop_price present and limit_price present -> STOP_LIMIT
                  - If stop_price present -> STOP
                  - If trail_amount or trail_percent present -> TRAIL_STOP (or TRAIL_STOP_LIMIT if limit_price present)
                  - Else if limit_price present -> LIMIT
                  - Else -> MARKET
              * Any legacy 'action' attribute is ignored for side; only quantity sign determines direction.
          - Parameters supported on the signal:
              limit_price (or price as alias), stop_price, trail_amount, trail_percent, trigger_on_range
          - Brackets/OCO:
              * If take_profit_price or stop_loss_* provided, broker creates child orders after parent fill.
              * Children are sized to filled quantity, inherit symbol and OCO-cancel each other.
              * stop child may be static stop, stop-limit (with stop_limit_price), or trailing (trail_amount/percent).
        """
        # Quantity sign determines side; do NOT read any 'action' attribute for side
        qty_in = float(signal.quantity)
        quantity = qty_in  # preserve sign as provided by caller

        # Extract optional params from Signal if present
        limit_price = getattr(signal, 'limit_price', None)
        if limit_price is None:
            # Back-compat: some code uses 'price' for limit
            limit_price = getattr(signal, 'price', None)
        stop_price = getattr(signal, 'stop_price', None)
        trail_amount = getattr(signal, 'trail_amount', None)
        trail_percent = getattr(signal, 'trail_percent', None)
        trigger_on_range = getattr(signal, 'trigger_on_range', True)

        # Bracket params from signal
        take_profit_price = getattr(signal, 'take_profit_price', None)
        stop_loss_price = getattr(signal, 'stop_loss_price', None)
        stop_limit_price = getattr(signal, 'stop_limit_price', None)  # optional limit for stop-limit child
        stop_trail_amount = getattr(signal, 'stop_trail_amount', None)
        stop_trail_percent = getattr(signal, 'stop_trail_percent', None)

        # Resolve order_type
        raw_type = getattr(signal, 'order_type', None)
        order_type: OrderType
        if raw_type:
            rt = str(raw_type).lower()
            if rt == 'market':
                order_type = OrderType.MARKET
            elif rt == 'limit':
                order_type = OrderType.LIMIT
            elif rt == 'stop':
                order_type = OrderType.STOP
            elif rt == 'stop_limit':
                order_type = OrderType.STOP_LIMIT
            elif rt == 'trail_stop':
                order_type = OrderType.TRAIL_STOP
            elif rt == 'trail_stop_limit':
                order_type = OrderType.TRAIL_STOP_LIMIT
            else:
                # Unknown text, fall back to inference below
                raw_type = None

        if not raw_type:
            # Infer from provided fields for backward-compat
            has_limit = limit_price is not None
            has_stop = stop_price is not None
            has_trail = (trail_amount is not None) or (trail_percent is not None)
            if has_trail and has_limit:
                order_type = OrderType.TRAIL_STOP_LIMIT
            elif has_trail:
                order_type = OrderType.TRAIL_STOP
            elif has_stop and has_limit:
                order_type = OrderType.STOP_LIMIT
            elif has_stop:
                order_type = OrderType.STOP
            elif has_limit:
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.MARKET

        order = Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            trigger_on_range=bool(trigger_on_range),
        )

        # Initialize trailing reference on creation if needed (will be updated upon first market data)
        order.trail_ref_price = None
        order.dynamic_stop = None
        order.triggered = False

        self.orders.append(order)

        # Attach bracket metadata to be materialized upon fill
        order._bracket_take_profit_price = float(take_profit_price) if take_profit_price is not None else None  # type: ignore[attr-defined]
        order._bracket_stop_loss_price = float(stop_loss_price) if stop_loss_price is not None else None  # type: ignore[attr-defined]
        order._bracket_stop_limit_price = float(stop_limit_price) if stop_limit_price is not None else None  # type: ignore[attr-defined]
        order._bracket_stop_trail_amount = float(stop_trail_amount) if stop_trail_amount is not None else None  # type: ignore[attr-defined]
        order._bracket_stop_trail_percent = float(stop_trail_percent) if stop_trail_percent is not None else None  # type: ignore[attr-defined]


        return order
    
    def execute_orders(self, market_data: Dict[str, pl.DataFrame], timestamp: datetime) -> List[Transaction]:
        """Execute pending orders based on current market data.

        Semantics:
          - Limit: fill if price is at/through limit in favor of the order
          - Stop / Stop-Limit: trigger when price crosses stop; then MARKET or LIMIT execution
          - Trail orders: update trail reference (high for long, low for short), recompute dynamic stop,
            trigger when cross; then MARKET or LIMIT based on type.
          - Trigger evaluation: by default use bar high/low range if available (open, high, low, close).
            If not available, use close only.
          - Gaps: if open crosses stop, treat as triggered at open and fill with slippage at first price.

        Returns:
          List of executed transactions
        """
        executed_transactions: List[Transaction] = []
 
        def bar_prices(df: pl.DataFrame) -> Tuple[float, float, float, float]:
            """
            Return (open, high, low, close) from DataFrame with sensible fallbacks.
            """
            c = float(df.item(-1, "close"))
            o = float(df.item(-1, "open")) if 'open' in df.columns else c
            h = float(df.item(-1, "high")) if 'high' in df.columns else max(c, o)
            l = float(df.item(-1, "low")) if 'low' in df.columns else min(c, o)
            return o, h, l, c

        for order in self.orders:
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                continue
            # Missing data for this symbol at this timestamp: defer instead of rejecting.
            if order.symbol not in market_data:
                # No snapshot for symbol -> keep order OPEN for future processing
                # Do not modify status from OPEN/PARTIALLY_FILLED
                continue

            symbol_data = market_data[order.symbol]
            # If we have an object but it's empty, also defer processing for this bar
            if hasattr(symbol_data, "is_empty") and symbol_data.is_empty():
                # Defer; do not reject
                continue

            o, h, l, c = bar_prices(symbol_data)
            is_buy = order.quantity > 0

            # Initialize trailing reference if needed
            if order.order_type in (OrderType.TRAIL_STOP, OrderType.TRAIL_STOP_LIMIT):
                if order.trail_ref_price is None:
                    order.trail_ref_price = c
                # Update reference: highest since activation for long, lowest for short
                if is_buy:
                    # For buy-to-cover (of a short), ref is the lowest low
                    order.trail_ref_price = min(order.trail_ref_price, l)
                else:
                    # For sell (of a long), ref is the highest high
                    order.trail_ref_price = max(order.trail_ref_price, h)
                # Compute dynamic stop
                if order.trail_amount is not None:
                    if is_buy:
                        order.dynamic_stop = order.trail_ref_price + float(order.trail_amount)
                    else:
                        order.dynamic_stop = order.trail_ref_price - float(order.trail_amount)
                else:
                    pct = float(order.trail_percent or 0.0)
                    if is_buy:
                        order.dynamic_stop = order.trail_ref_price * (1.0 + pct)
                    else:
                        order.dynamic_stop = order.trail_ref_price * (1.0 - pct)

            # Determine trigger price for stop-type orders
            stop_level = None
            if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                stop_level = float(order.stop_price)
            elif order.order_type in (OrderType.TRAIL_STOP, OrderType.TRAIL_STOP_LIMIT):
                stop_level = float(order.dynamic_stop) if order.dynamic_stop is not None else None

            # Evaluate triggers
            def crossed() -> bool:
                if stop_level is None:
                    return False
                if order.trigger_on_range:
                    # Use high/low intrabar
                    if is_buy:
                        return h >= stop_level
                    else:
                        return l <= stop_level
                else:
                    # Close only
                    return (c >= stop_level) if is_buy else (c <= stop_level)

            # Determine if order is active-to-fill this bar
            should_attempt_fill = False
            effective_order_type = order.order_type

            if order.order_type == OrderType.MARKET:
                should_attempt_fill = True
            elif order.order_type == OrderType.LIMIT:
                # Range-aware limit evaluation when trigger_on_range is True
                lim = float(order.limit_price)
                if order.trigger_on_range:
                    if (is_buy and l <= lim) or (not is_buy and h >= lim):
                        should_attempt_fill = True
                else:
                    if (is_buy and c <= lim) or (not is_buy and c >= lim):
                        should_attempt_fill = True
            elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAIL_STOP, OrderType.TRAIL_STOP_LIMIT):
                if not order.triggered and crossed():
                    order.triggered = True
                    # For gaps: choose first available price consistent with direction
                    # Later when computing execution price, we will apply slippage
                if order.triggered:
                    if order.order_type in (OrderType.STOP, OrderType.TRAIL_STOP):
                        effective_order_type = OrderType.MARKET
                        should_attempt_fill = True
                    else:
                        effective_order_type = OrderType.LIMIT
                        # For trailing stop limit: if limit missing, default to stop_level to create a peg
                        lim = order.limit_price if order.limit_price is not None else stop_level
                        order.limit_price = lim
                        if (is_buy and c <= float(lim)) or (not is_buy and c >= float(lim)):
                            should_attempt_fill = True

            if not should_attempt_fill:
                continue

            # Determine base price to execute against (pre-slippage)
            base_price = c
            if effective_order_type == OrderType.MARKET:
                # If triggered due to gap, approximate fill at open; else use close
                if order.triggered and order.trigger_on_range and ((is_buy and o >= (stop_level or -float('inf'))) or (not is_buy and o <= (stop_level or float('inf')))):
                    base_price = o
                else:
                    base_price = c
            elif effective_order_type == OrderType.LIMIT:
                # Execute at the prevailing market price but only if favorable to limit
                # This preserves realistic limit order behavior (price improvement allowed).
                base_price = c

            # Apply slippage to execution price
            exec_price = self.slippage_model.apply_slippage(base_price, order.quantity, symbol_data, is_buy)

            # Commission and affordability
            unfilled = order.quantity - order.filled_quantity
            trade_value = unfilled * exec_price
            commission_cost = self.commission_model.calculate_commission(exec_price, unfilled)
            if is_buy and (trade_value + commission_cost > self.cash):
                max_affordable = self.cash / (exec_price + (commission_cost / max(abs(unfilled), 1e-9)))
                if max_affordable <= 0:
                    continue
                unfilled = min(unfilled, max_affordable)
                trade_value = unfilled * exec_price
                commission_cost = self.commission_model.calculate_commission(exec_price, unfilled)

            # Execute
            transaction = Transaction(
                id=str(uuid.uuid4()),
                order_id=order.id,
                symbol=order.symbol,
                quantity=unfilled,
                price=exec_price,
                commission=commission_cost,
                timestamp=timestamp
            )

            # Update order status
            prev_filled = order.filled_quantity
            order.filled_quantity += unfilled
            order.filled_price = (
                ((order.filled_price or 0.0) * prev_filled) + (exec_price * unfilled)
            ) / order.filled_quantity if order.filled_quantity > 0 else exec_price

            if abs(order.filled_quantity - order.quantity) < 1e-6:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

            # Update position and cash
            self._update_position(transaction.symbol, transaction.quantity, transaction.price)
            self.cash -= (trade_value + commission_cost)

            # Record transaction
            self.transactions.append(transaction)
            executed_transactions.append(transaction)

            # OCO logic: if this order is part of an OCO group and is (fully) filled, cancel siblings
            if order.oco_group_id and order.status == OrderStatus.FILLED:
                for other in self.orders:
                    if other.id != order.id and other.oco_group_id == order.oco_group_id:
                        if other.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                            other.status = OrderStatus.CANCELLED

            # Bracket materialization: when a parent fills for the first time, create children
            if hasattr(order, "_bracket_take_profit_price") or hasattr(order, "_bracket_stop_loss_price"):
                # Only create children when parent is fully filled (common behavior)
                if order.status == OrderStatus.FILLED:
                    tp = getattr(order, "_bracket_take_profit_price", None)
                    sl = getattr(order, "_bracket_stop_loss_price", None)
                    sll = getattr(order, "_bracket_stop_limit_price", None)
                    trl_amt = getattr(order, "_bracket_stop_trail_amount", None)
                    trl_pct = getattr(order, "_bracket_stop_trail_percent", None)

                    if tp is not None or sl is not None or trl_amt is not None or trl_pct is not None:
                        oco_id = str(uuid.uuid4())
                        child_qty = -order.quantity  # opposite side to exit position

                        # Take-profit child (limit)
                        if tp is not None:
                            tp_order = Order(
                                id=str(uuid.uuid4()),
                                symbol=order.symbol,
                                quantity=child_qty,
                                order_type=OrderType.LIMIT,
                                limit_price=float(tp),
                                trigger_on_range=True,
                            )
                            tp_order.parent_id = order.id
                            tp_order.oco_group_id = oco_id
                            tp_order.bracket_role = "take_profit"
                            self.orders.append(tp_order)

                        # Stop child: static/stop-limit or trailing
                        if trl_amt is not None or trl_pct is not None:
                            # Trailing stop or trailing stop-limit (if sll provided)
                            ot = OrderType.TRAIL_STOP_LIMIT if sll is not None else OrderType.TRAIL_STOP
                            stop_child = Order(
                                id=str(uuid.uuid4()),
                                symbol=order.symbol,
                                quantity=child_qty,
                                order_type=ot,
                                limit_price=float(sll) if sll is not None else None,
                                trail_amount=float(trl_amt) if trl_amt is not None else None,
                                trail_percent=float(trl_pct) if trl_pct is not None else None,
                                trigger_on_range=True,
                                # Correctly seed trail_ref_price from parent fill
                                trail_ref_price=order.filled_price,
                            )
                        elif sl is not None:
                            # Static stop or stop-limit
                            ot = OrderType.STOP_LIMIT if sll is not None else OrderType.STOP
                            stop_child = Order(
                                id=str(uuid.uuid4()),
                                symbol=order.symbol,
                                quantity=child_qty,
                                order_type=ot,
                                stop_price=float(sl),
                                limit_price=float(sll) if sll is not None else None,
                                trigger_on_range=True,
                            )
                        else:
                            stop_child = None

                        if stop_child is not None:
                            stop_child.parent_id = order.id
                            stop_child.oco_group_id = oco_id
                            stop_child.bracket_role = "stop_loss"
                            self.orders.append(stop_child)

        return executed_transactions

    # Convenience path to ensure all market data access can go through DataPortal
    def execute_orders_with_portal(self, symbols: List[str], timestamp: datetime, market_hours: Optional[tuple] = None) -> List[Transaction]:
        """
        Fetch the per-timestamp snapshot via DataPortal and execute orders.
        """
        try:
            dp = self._data_portal
            if dp is None:
                raise RuntimeError("DataPortal is not set on Broker. Pass it into Broker(...) at initialization.")
            ts_slice = dp.get_slice_for_timestamp(timestamp, sorted(list(symbols)), market_hours=market_hours)
        except Exception:
            ts_slice = pl.DataFrame()
        # Build symbol->DataFrame map
        market_data_dict: Dict[str, pl.DataFrame] = {}
        if not ts_slice.is_empty() and 'symbol' in ts_slice.columns:
            s = ts_slice.get_column('symbol')
            if s.dtype != pl.String:
                ts_slice = ts_slice.with_columns(pl.col('symbol').cast(pl.Utf8))
            for (sym,), grp in ts_slice.group_by('symbol', maintain_order=True):
                market_data_dict[str(sym)] = grp
        return self.execute_orders(market_data_dict, timestamp)
    
    def _update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position for a symbol.
        
        Args:
            symbol: Trading symbol
            quantity: Transaction quantity
            price: Transaction price
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        self.positions[symbol].update(quantity, price)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if order was cancelled, False if not found or not open
        """
        for order in self.orders:
            if order.id == order_id and order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(symbol)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        for order in self.orders:
            if order.id == order_id:
                return order
        return None
    
    def update_market_values(self, market_data: Dict[str, pl.DataFrame]) -> None:
        """Update market values for all positions.
        
        Args:
            market_data: Current market data by symbol
        """
        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].is_empty():
                price = market_data[symbol].item(-1, "close")
                position.update_market_value(price)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_account(self) -> Dict[str, Any]:
        """Get account summary.
        
        Returns:
            Account summary dictionary
        """
        return {
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'cost_basis': pos.cost_basis,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl()
            } for symbol, pos in self.positions.items() if abs(pos.quantity) > 1e-6},
            'portfolio_value': self.get_portfolio_value(),
        } 
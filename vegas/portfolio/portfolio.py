"""Portfolio layer for the Vegas backtesting engine.

This module provides a portfolio tracking system for event-driven backtesting.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import polars as pl

SHORT_INITIAL_MARGIN_RATE = 0.50  # Reg T approximation: 50% initial margin on short market value
EPS = 1e-6


class Position:
    """A class representing a portfolio position (long or short)."""
    def __init__(self, symbol: str, quantity: float, value: float = 0.0):
        self.symbol = symbol
        self.quantity = quantity  # negative quantity represents a short
        self.value = value

    def __str__(self):
        return f"Position({self.symbol}, {self.quantity}, ${self.value:.2f})"

    def __repr__(self):
        return self.__str__()


class Portfolio:
    """Portfolio tracking system for the Vegas event-driven backtesting engine."""
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, float] = {}        # symbol -> quantity (can be negative for shorts)
        # Enriched per-position dict keyed by symbol. Each value:
        # {'symbol','buy_price','qty','current_price','current_market_value','pnl','pnl_pct'}
        self.position_values: Dict[str, Dict[str, float]] = {}
        # Average entry price (moving average) for current net position per symbol
        self.avg_price: Dict[str, float] = {}
        self.current_equity = initial_capital

        # Whether this portfolio is seeded from a live broker snapshot.
        # When set, initial_capital remains as a reference but not used to assert returns in live mode.
        self._seeded_from_snapshot = False

        # Margin and buying power tracking
        self.short_margin_requirement: float = 0.0  # Total margin requirement for shorts
        self.buying_power: float = initial_capital  # Available to initiate long or short considering margin

        # History
        self.equity_history = []      # dicts: timestamp, equity, cash, buying_power
        self.position_history = []    # dicts: timestamp, symbol, quantity, value
        self.transaction_history = [] # dicts: timestamp, symbol, quantity, price, commission

        # In-memory cache of last known prices per symbol (updated only from price_lookup values)
        self._last_price: Dict[str, float] = {}

        self._logger = logging.getLogger('vegas.portfolio')
        # Optional DataPortal for price lookups
        self._data_portal = None

    def set_data_portal(self, data_portal) -> None:
        """Inject a DataPortal for price lookups and mark-to-market operations."""
        self._data_portal = data_portal

    def set_account_snapshot(self, cash: float, positions_dict: Dict[str, Dict[str, float]]) -> None:
        """
        Seed the portfolio state from an external brokerage account snapshot.

        positions_dict format:
          {
            "AAPL": {"quantity": 10.0, "avg_price": 180.0},
            "MSFT": {"quantity": -5.0, "avg_price": 410.0},
            ...
          }

        - Sets current_cash to the provided cash
        - Sets positions and avg_price using the provided snapshot
        - Recomputes equity using last-known prices (falls back to avg_price if no price yet)
        - Marks portfolio as seeded from snapshot
        """
        try:
            self.current_cash = float(cash)
        except Exception:
            self.current_cash = cash

        self.positions = {}
        self.avg_price = {}
        self.position_values = {}
        self.transaction_history = []
        self.position_history = []
        self.equity_history = []

        # Load positions and average prices from snapshot
        for sym, info in positions_dict.items():
            qty = float(info.get("quantity", 0.0))
            if abs(qty) < EPS:
                continue
            avg_px = float(info.get("avg_price", 0.0))
            self.positions[sym] = qty
            self.avg_price[sym] = abs(avg_px)

        # With no market data at snapshot time, approximate equity by cash + sum(qty*avg_price)
        total_position_value = 0.0
        for sym, qty in self.positions.items():
            px = float(self.avg_price.get(sym, 0.0))
            total_position_value += qty * px

        self.current_equity = self.current_cash + total_position_value
        self._seeded_from_snapshot = True

        # Record a synthetic equity snapshot with timestamp=epoch 0 to indicate seeding point will be recorded on first bar
        # We avoid adding a timestamped equity row here to let engine add the first bar state.

    def _recompute_short_margin_and_buying_power(self):
        """Recompute short margin requirement and buying power using Reg T approximation.

        Reg T short initial margin: 150% of current short market value.
        Requirement = 1.5 * sum(|qty| * price) for all short positions.
        Buying power = current_cash + long_market_value - requirement.
        """
        total_short_market_value = 0.0
        total_long_market_value = 0.0
        for symbol, pv in self.position_values.items():
            # pv is enriched dict; use current_market_value
            value = pv.get('current_market_value', 0.0)
            if value < 0:
                total_short_market_value += abs(value)
            else:
                total_long_market_value += value

        requirement = 1.5 * total_short_market_value
        self.short_margin_requirement = requirement
        self.buying_power = self.current_cash + total_long_market_value - self.short_margin_requirement

    def _adjust_for_buying_power(self, symbol: str, quantity: float, price: float, commission: float) -> float:
        """Adjust order quantity to respect buying power with Reg T margin for shorts.

        For longs: require cash >= qty*price + commission (existing approach).
        For shorts: require buying_power >= delta_requirement where
          delta_requirement = 1.5 * (abs(new_short_mv) - abs(old_short_mv_for_symbol))
        """
        if quantity > 0:
            # Long buy check against cash
            cost = quantity * price + commission
            if cost <= self.current_cash + EPS:
                return quantity
            max_qty = max(0.0, (self.current_cash - commission) / price)
            if max_qty < EPS:
                self._logger.warning(
                    f"Skipping buy of {quantity} {symbol} - insufficient cash "
                    f"(needed: ${cost:.2f}, available: ${self.current_cash:.2f})"
                )
                return 0.0
            self._logger.warning(
                f"Adjusted buy from {quantity} to {max_qty} shares of {symbol} due to insufficient cash"
            )
            return max_qty

        # Selling: can be closing long or initiating/increasing short
        current_qty = self.positions.get(symbol, 0.0)
        proposed_qty = current_qty + quantity  # quantity is negative for sell

        if proposed_qty >= -EPS:
            # Not creating/increasing short exposure; either reducing long or flat -> allow
            return quantity

        # We are initiating/increasing short. Compute delta requirement.
        old_short_mv = abs(min(0.0, current_qty) * price)   # current short notional for this symbol
        new_short_mv = abs(min(0.0, proposed_qty) * price)  # proposed short notional for this symbol
        delta_req = 1.5 * max(0.0, new_short_mv - old_short_mv)

        # Ensure buying power covers increased requirement
        # Use up-to-date values by recomputing first
        self._recompute_short_margin_and_buying_power()

        if delta_req <= self.buying_power + EPS:
            return quantity

        # Need to scale sell size (more negative) so that delta_req fits into buying_power.
        # Let x = abs(additional short shares) = abs(proposed_qty_short - current_qty_short)
        # delta_req = 1.5 * x * price <= buying_power -> x <= buying_power / (1.5 * price)
        current_short_shares = abs(min(0.0, current_qty))
        target_additional = max(0.0, (self.buying_power) / (1.5 * price))
        max_total_short_shares = current_short_shares + target_additional
        max_proposed_qty = -max_total_short_shares  # negative

        adjusted_qty = max_proposed_qty - current_qty  # this is negative
        if abs(adjusted_qty) < EPS:
            self._logger.warning(
                f"Skipping short sell of {abs(quantity)} {symbol} - insufficient buying power "
                f"(delta requirement: ${delta_req:.2f}, buying_power: ${self.buying_power:.2f})"
            )
            return 0.0

        self._logger.warning(
            f"Adjusted short sell from {abs(quantity)} to {abs(adjusted_qty)} shares of {symbol} due to "
            f"insufficient buying power (needed Δreq: ${delta_req:.2f}, available: ${self.buying_power:.2f})"
        )
        return adjusted_qty

    def update_from_transactions(self, timestamp, transactions, market_data: Optional[pl.DataFrame] = None):
        """Update portfolio based on executed transactions using DataPortal for pricing.

        Notes:
        - The `market_data` parameter is ignored; all pricing comes from the injected DataPortal.
        - `transactions` has columns: symbol, quantity, price, commission.
        - Allows short-selling with Reg T 150% initial margin.
        """

        def _format_trade_message(trade):
            side = "BOT" if trade['quantity'] > 0 else "SLD"
            abs_qty = abs(trade['quantity'])
            msg = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} {side} {abs_qty} {trade['symbol']} @ ${trade['price']:.2f} Commission ${trade['commission']:.2f} "
            return msg

        # Process transactions
        if not transactions.is_empty():
            for txn in transactions.to_dicts():
                symbol = txn['symbol']
                quantity = float(txn['quantity'])
                price = float(txn['price'])
                commission = float(txn.get('commission', 0.0))

                try:
                    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_str = str(timestamp)

                # Validate and adjust for cash/buying power
                adjusted_qty = self._adjust_for_buying_power(symbol, quantity, price, commission)
                if abs(adjusted_qty) < EPS:
                    continue
                quantity = adjusted_qty

                # Update cash: buy reduces cash; sell increases cash
                # Note: commission always reduces cash
                self.current_cash -= (quantity * price + commission)

                # Update positions and average price
                current_qty = self.positions.get(symbol, 0.0)
                new_qty = current_qty + quantity

                if abs(new_qty) < EPS:
                    # Position closed
                    if symbol in self.positions:
                        del self.positions[symbol]
                    if symbol in self.avg_price:
                        del self.avg_price[symbol]
                else:
                    # Determine if same side, reducing, or crossing
                    if abs(current_qty) < EPS:
                        # Opening new position
                        self.positions[symbol] = new_qty
                        self.avg_price[symbol] = abs(price)
                    else:
                        same_side = (current_qty > 0 and new_qty > 0) or (current_qty < 0 and new_qty < 0)
                        increasing_magnitude = abs(new_qty) > abs(current_qty)
                        if same_side and increasing_magnitude:
                            # Adding to existing position -> moving average
                            old_avg = self.avg_price.get(symbol, abs(price))
                            # Use share counts by magnitude (works for long/short)
                            new_avg = (old_avg * abs(current_qty) + abs(price) * abs(quantity)) / abs(new_qty)
                            self.positions[symbol] = new_qty
                            self.avg_price[symbol] = new_avg
                        elif same_side and not increasing_magnitude:
                            # Reducing position on same side -> avg unchanged
                            self.positions[symbol] = new_qty
                            # avg_price remains
                        else:
                            # Crossing zero: close old side, open new side remainder at current trade price
                            # Example: long 10, sell 15 -> new short 5 at price
                            self.positions[symbol] = new_qty
                            self.avg_price[symbol] = abs(price)
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'commission': commission
                }
                # Record transaction
                self.transaction_history.append(trade)
                msg = _format_trade_message(trade)
                self._logger.info(msg)

        # Update position values using DataPortal-derived prices only
        self.position_values = {}
        total_position_value = 0.0
        
        # Build a lookup of current prices for all open position symbols via DataPortal
        price_lookup: Dict[str, float] = {}
        if self._data_portal is not None:
            for symbol in list(self.positions.keys()):
                try:
                    px = self._data_portal.get_spot_value(symbol, 'close', timestamp)
                except Exception:
                    px = None
                if px is not None:
                    price_lookup[symbol] = float(px)
                    self._last_price[symbol] = float(px)

        for symbol, qty in self.positions.items():
            # Resolve current price with precedence: portal lookup -> last known -> 0.0
            if symbol in price_lookup:
                current_price = float(price_lookup[symbol])
            else:
                current_price = float(self._last_price.get(symbol, 0.0))

            current_market_value = qty * current_price
            buy_price = self.avg_price.get(symbol, 0.0)

            # P&L based on net position sign
            pnl = (current_price - buy_price) * qty
            denom = abs(buy_price) * abs(qty)
            pnl_pct = (pnl / denom) * 100 if denom > EPS else 0.0

            self.position_values[symbol] = {
                'symbol': symbol,
                'buy_price': buy_price,
                'qty': qty,
                'current_price': current_price,
                'current_market_value': current_market_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            total_position_value += current_market_value

            # Record position snapshot
            self.position_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': qty,
                'value': current_market_value
            })

        # Recompute margin and buying power
        self._recompute_short_margin_and_buying_power()

        # Update equity: equity = cash + sum(positions)
        self.current_equity = self.current_cash + total_position_value

        # Record equity history
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': self.current_equity,
            'cash': self.current_cash,
            'buying_power': self.buying_power,
            'short_margin_requirement': self.short_margin_requirement
        })

    def get_portfolio_value(self):
        """Get the current portfolio value (equity)."""
        return self.current_equity

    def get_equity_curve(self):
        """Get the portfolio equity curve DataFrame."""
        if not self.equity_history:
            return pl.DataFrame(schema={
                'timestamp': pl.Datetime('ns'),
                'equity': pl.Float64,
                'cash': pl.Float64,
                'buying_power': pl.Float64,
                'short_margin_requirement': pl.Float64
            })
        return pl.DataFrame(self.equity_history)

    def get_returns(self):
        """Get portfolio returns."""
        equity_curve = self.get_equity_curve()
        if len(equity_curve) <= 1:
            return pl.DataFrame(schema={
                'timestamp': pl.Datetime('ns'),
                'return': pl.Float64,
                'cumulative_return': pl.Float64,
            })

        equity_curve = equity_curve.with_columns(
            (pl.col('equity').pct_change().fill_null(0)).alias('return')
        ).with_columns(
            (1 + pl.col('return')).cum_prod().alias('cumulative_return')
        )

        return equity_curve[['timestamp', 'return', 'cumulative_return']]

    def get_transactions(self):
        """Get all transactions as DataFrame."""
        if not self.transaction_history:
            return pl.DataFrame(schema={
                'timestamp': pl.Datetime('ns'),
                'symbol': pl.String,
                'quantity': pl.Float64,
                'price': pl.Float64,
                'commission': pl.Float64
            })
        return pl.DataFrame(self.transaction_history)

    def get_positions(self):
        """Get current portfolio positions as Position objects."""
        out = []
        for symbol, qty in self.positions.items():
            if abs(qty) > EPS:
                pv = self.position_values.get(symbol, {})
                value = pv.get('current_market_value', 0.0)
                out.append(Position(symbol, qty, value))
        return out

    def get_positions_dataframe(self):
        """Get current positions as DataFrame."""
        rows = []
        for symbol, qty in self.positions.items():
            pv = self.position_values.get(symbol, {})
            value = pv.get('current_market_value', 0.0)
            rows.append({
                'symbol': symbol,
                'quantity': qty,
                'value': value,
                'weight': (value / self.current_equity) if self.current_equity != 0 else 0.0
            })
        if not rows:
            return pl.DataFrame(schema={
                'symbol': pl.String,
                'quantity': pl.Float64,
                'value': pl.Float64,
                'weight': pl.Float64
            })
        return pl.DataFrame(rows)

    def get_positions_history(self):
        """Get history of positions grouped by timestamp."""
        history = {}
        df = pl.DataFrame(self.position_history)
        if not df.is_empty():
            for ts, group in df.group_by('timestamp'):
                history[ts] = {
                    row['symbol']: {'quantity': row['quantity'], 'value': row['value']}
                    for row in group.iter_rows(named=True)
                }
        return history

    def get_stats(self):
        """Calculate basic performance statistics."""
        stats: Dict[str, Any] = {}
        stats['initial_capital'] = self.initial_capital
        stats['final_value'] = self.current_equity
        stats['cash'] = self.current_cash
        stats['buying_power'] = self.buying_power
        stats['short_margin_requirement'] = self.short_margin_requirement
        stats['total_return'] = self.current_equity - self.initial_capital
        stats['total_return_pct'] = (self.current_equity / self.initial_capital - 1) * 100

        transactions_df = self.get_transactions()
        stats['num_trades'] = len(transactions_df)

        equity_df = self.get_equity_curve()
        returns_df = self.get_returns()

        if len(equity_df) > 1:
            equity_df = equity_df.with_columns(
                pl.col('equity').cum_max().alias('previous_peak')
            ).with_columns(
                ((pl.col('equity') - pl.col('previous_peak')) / pl.col('previous_peak') * 100).alias('drawdown')
            )
            max_drawdown = equity_df.select(pl.col('drawdown').min()).item()
            stats['max_drawdown_pct'] = abs(max_drawdown)

            if returns_df.height > 1:
                daily_returns = returns_df['return']
                mean_return = daily_returns.mean()
                std_return = daily_returns.std()
                if std_return and std_return > 0:
                    stats['sharpe_ratio'] = (mean_return / std_return) * (252 ** 0.5)
                    stats['annual_return_pct'] = ((1 + mean_return) ** 252 - 1) * 100
                else:
                    stats['sharpe_ratio'] = 0.0
                    stats['annual_return_pct'] = 0.0
            else:
                stats['sharpe_ratio'] = 0.0
                stats['annual_return_pct'] = 0.0

        return stats

    def build_positions_ledger(self, latest_price_lookup: Optional[Dict[str, float]] = None) -> pl.DataFrame:
        """
        Build and return a complete positions ledger (open and closed) as a Polars DataFrame.

        The ledger reconstructs position lifecycles from the recorded transaction_history and
        position/equity snapshots, producing one row per realized or still-open position.

        Columns:
          - symbol: str
          - pos_open_ts: datetime
          - pos_close_ts: datetime or None (if still open)
          - side: 'LONG' or 'SHORT'
          - qty_open: float (absolute number of shares at open)
          - qty_close: float (absolute number of shares closed; equals qty_open for fully closed)
          - buy_price: float (average entry price for the net position)
          - exit_price: float or None (for open positions, uses last known/lookup price)
          - realized_pnl: float (only for closed positions)
          - unrealized_pnl: float (only for open positions)
          - pnl_dollars: float (realized for closed, mark-to-market for open)
          - pnl_pct: float (% relative to notional |qty|*buy_price)
          - commissions: float (sum of commissions across all trades in the position window)

        Note:
          - This reconstruction assumes FIFO at the net-position level, matching how avg_price is tracked.
          - For open positions without a current price available, exit_price will be None and P&L = 0.0.

        Args:
          latest_price_lookup: Optional mapping symbol->last price to value open positions.

        Returns:
          Polars DataFrame with the ledger rows (possibly empty).
        """
        # Fast return if no transactions ever recorded
        if not self.transaction_history:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "pos_open_ts": pl.Datetime("ns"),
                "pos_close_ts": pl.Datetime("ns"),
                "side": pl.Utf8,
                "qty_open": pl.Float64,
                "qty_close": pl.Float64,
                "buy_price": pl.Float64,
                "exit_price": pl.Float64,
                "realized_pnl": pl.Float64,
                "unrealized_pnl": pl.Float64,
                "pnl_dollars": pl.Float64,
                "pnl_pct": pl.Float64,
                "commissions": pl.Float64,
            })

        # Group transactions by symbol in timestamp order
        tx_df = pl.DataFrame(self.transaction_history)
        tx_df = tx_df.sort("timestamp")

        rows: List[Dict[str, Any]] = []
        latest_price_lookup = latest_price_lookup or {}

        # Helper to finalize and append a ledger row
        def _append_row(symbol: str,
                        open_ts: datetime,
                        close_ts: Optional[datetime],
                        side: str,
                        qty_open: float,
                        qty_close: float,
                        buy_price: float,
                        exit_price: Optional[float],
                        commissions: float) -> None:
            notional = abs(qty_open) * abs(buy_price)
            if exit_price is None:
                pnl = 0.0
                pnl_pct = 0.0
            else:
                # P&L sign depends on side and direction
                sign = 1.0 if side == "LONG" else -1.0
                pnl = sign * (exit_price - buy_price) * abs(qty_close)
                pnl_pct = (pnl / notional * 100.0) if notional > EPS else 0.0

            realized = 0.0
            unrealized = 0.0
            if close_ts is None:
                # open position -> unrealized
                unrealized = float(pnl)
            else:
                realized = float(pnl)

            rows.append({
                "symbol": symbol,
                "pos_open_ts": open_ts,
                "pos_close_ts": close_ts,
                "side": side,
                "qty_open": float(qty_open),
                "qty_close": float(qty_close),
                "buy_price": float(buy_price),
                "exit_price": None if exit_price is None else float(exit_price),
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "pnl_dollars": float(pnl),
                "pnl_pct": float(pnl_pct),
                "commissions": float(commissions),
            })

        # Reconstruct per-symbol lifecycles by walking quantity over time
        for symbol, sym_tx in tx_df.group_by("symbol", maintain_order=True):
            # state for current net position
            net_qty = 0.0
            avg_px = 0.0
            pos_open_ts: Optional[datetime] = None
            side: Optional[str] = None
            accrued_commission = 0.0

            for tx in sym_tx.iter_rows(named=True):
                ts: datetime = tx["timestamp"]
                qty: float = float(tx["quantity"])
                px: float = float(tx["price"])
                com: float = float(tx.get("commission", 0.0) or 0.0)
                accrued_commission += com

                prev_qty = net_qty
                new_qty = prev_qty + qty

                if abs(prev_qty) < EPS:
                    # Opening a new net position
                    net_qty = new_qty
                    if abs(net_qty) < EPS:
                        # Degenerate open/close same tick; treat as no-op
                        continue
                    side = "LONG" if net_qty > 0 else "SHORT"
                    avg_px = abs(px)
                    pos_open_ts = ts
                    continue

                # Same side?
                prev_side_long = prev_qty > 0
                new_side_long = new_qty > 0
                same_side = (prev_side_long and new_side_long) or ((not prev_side_long) and (not new_side_long))

                if same_side:
                    # increasing or reducing on same side
                    if abs(new_qty) > abs(prev_qty):
                        # adding -> moving average
                        avg_px = (abs(avg_px) * abs(prev_qty) + abs(px) * abs(qty)) / abs(new_qty)
                        net_qty = new_qty
                    else:
                        # reducing but still same side -> keep avg_px
                        net_qty = new_qty
                else:
                    # Crossing through zero: close previous and open new residual
                    # Close leg uses the trade price 'px' as exit
                    close_qty = abs(prev_qty)
                    _append_row(
                        symbol=symbol,
                        open_ts=pos_open_ts if pos_open_ts is not None else ts,
                        close_ts=ts,
                        side="LONG" if prev_qty > 0 else "SHORT",
                        qty_open=abs(prev_qty),
                        qty_close=close_qty,
                        buy_price=avg_px,
                        exit_price=abs(px),
                        commissions=accrued_commission
                    )
                    # Start a new position with the remainder
                    residual = new_qty  # signed
                    if abs(residual) < EPS:
                        # fully flat after crossing; reset state
                        net_qty = 0.0
                        avg_px = 0.0
                        pos_open_ts = None
                        side = None
                        accrued_commission = 0.0
                    else:
                        net_qty = residual
                        side = "LONG" if net_qty > 0 else "SHORT"
                        avg_px = abs(px)
                        pos_open_ts = ts
                        accrued_commission = 0.0  # commissions after close attributed to next cycle

            # End of symbol transactions: if still open, mark-to-market with last known price
            if abs(net_qty) > EPS and pos_open_ts is not None and side is not None:
                # Resolve exit/mark price: prefer provided lookup, else DataPortal at last equity ts, else last known cache, else None
                exit_px: Optional[float] = None
                if symbol in latest_price_lookup:
                    exit_px = float(latest_price_lookup[symbol])
                else:
                    # Try DataPortal at the most recent equity snapshot timestamp
                    last_ts: Optional[datetime] = None
                    try:
                        if self.equity_history:
                            last_ts = self.equity_history[-1]['timestamp']
                        elif self.transaction_history:
                            last_ts = self.transaction_history[-1]['timestamp']
                    except Exception:
                        last_ts = None
                    if self._data_portal is not None and last_ts is not None:
                        try:
                            spot = self._data_portal.get_spot_value(symbol, 'close', last_ts)
                            if spot is not None:
                                exit_px = float(spot)
                        except Exception:
                            exit_px = None
                    if exit_px is None:
                        exit_px = float(self._last_price.get(symbol)) if symbol in self._last_price else None

                _append_row(
                    symbol=symbol,
                    open_ts=pos_open_ts,
                    close_ts=None,
                    side=side,
                    qty_open=abs(net_qty),
                    qty_close=abs(net_qty),
                    buy_price=avg_px,
                    exit_price=exit_px,
                    commissions=accrued_commission
                )

        if not rows:
            return pl.DataFrame(schema={
                "symbol": pl.Utf8,
                "pos_open_ts": pl.Datetime("ns"),
                "pos_close_ts": pl.Datetime("ns"),
                "side": pl.Utf8,
                "qty_open": pl.Float64,
                "qty_close": pl.Float64,
                "buy_price": pl.Float64,
                "exit_price": pl.Float64,
                "realized_pnl": pl.Float64,
                "unrealized_pnl": pl.Float64,
                "pnl_dollars": pl.Float64,
                "pnl_pct": pl.Float64,
                "commissions": pl.Float64,
            })

        df = pl.DataFrame(rows)

        # Normalize/ensure scalar dtypes; guard against any list-typed columns due to mixed inputs
        # Cast columns only if their dtype is not already correct and not a List.
        safe_exprs = []
        for col, dtype in [
            ("symbol", pl.Utf8),
            ("pos_open_ts", pl.Datetime("ns")),
            ("pos_close_ts", pl.Datetime("ns")),
            ("side", pl.Utf8),
            ("qty_open", pl.Float64),
            ("qty_close", pl.Float64),
            ("buy_price", pl.Float64),
            ("exit_price", pl.Float64),
            ("realized_pnl", pl.Float64),
            ("unrealized_pnl", pl.Float64),
            ("pnl_dollars", pl.Float64),
            ("pnl_pct", pl.Float64),
            ("commissions", pl.Float64),
        ]:
            if col in df.columns:
                c = df.get_column(col)
                # If column accidentally ended up as a List type, collapse lists by taking first non-null element
                if isinstance(c.dtype, pl.List):
                    df = df.with_columns(pl.col(col).list.first().alias(col))
                # After ensuring scalar, cast to target dtype
                safe_exprs.append(pl.col(col).cast(dtype))

        if safe_exprs:
            df = df.with_columns(safe_exprs)

        df = df.sort(["symbol", "pos_open_ts", "pos_close_ts"])

        return df
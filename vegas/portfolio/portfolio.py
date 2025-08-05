"""Portfolio layer for the Vegas backtesting engine.

This module provides a portfolio tracking system for event-driven backtesting.
"""
from typing import Dict, Any
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

    def update_from_transactions(self, timestamp, transactions, market_data):
        """Update portfolio based on executed transactions and current market data.

        Transactions: columns: symbol, quantity, price, commission
        Allows short-selling with Reg T 150% initial margin.
        """

        def _format_trade_message(trade):
            side = "BOT" if trade['quantity'] > 0 else "SLD"
            abs_qty = abs(trade['quantity'])
            msg = f"{side} {abs_qty} {trade['symbol']} @ ${trade['price']:.2f} Commission ${trade['commission']:.2f} "
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

        # Update position values based on current market data
        self.position_values = {}
        total_position_value = 0.0

        price_lookup = {}
        if 'symbol' in market_data.columns and 'close' in market_data.columns:
            price_lookup = {row['symbol']: float(row['close']) for row in market_data.select(['symbol', 'close']).to_dicts()}

        # Opportunistically update last-known cache from available market data prices
        # Only learn from explicit price_lookup values as specified
        for sym, px in price_lookup.items():
            if px is not None:
                self._last_price[sym] = float(px)

        for symbol, qty in self.positions.items():
            # Resolve current price with precedence: price_lookup -> last known -> 0.0
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
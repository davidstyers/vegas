"""
Example strategy demonstrating bracket orders with stop losses and take profits.

This strategy shows how the on_trade callback captures all edge cases including:
- Regular entry trades
- Take profit executions (bracket orders)
- Stop loss executions (bracket orders)
- Trailing stop orders
- OCO (One-Cancels-Other) orders

The strategy implements a simple momentum strategy with automatic risk management
through bracket orders, and demonstrates how to track and respond to each type
of trade execution.
"""

from datetime import datetime
from typing import List
import polars as pl

from vegas.strategy import Strategy, Signal, Context


class BracketOrderStrategy(Strategy):
    """Strategy demonstrating comprehensive bracket order handling with on_trade callbacks."""
    
    def initialize(self, context: Context) -> None:
        """Initialize strategy parameters."""
        context.symbol = "AAPL"
        context.momentum_period = 14
        context.entry_threshold = 0.02  # 2% momentum threshold
        
        # Risk management parameters
        context.stop_loss_pct = 0.03    # 3% stop loss
        context.take_profit_pct = 0.06  # 6% take profit (2:1 risk/reward)
        context.trailing_stop_pct = 0.02  # 2% trailing stop
        context.max_position_size = 1000
        
        # Trade tracking
        context.trade_count = 0
        context.entry_trades = 0
        context.take_profit_trades = 0
        context.stop_loss_trades = 0
        context.trailing_stop_trades = 0
        context.total_pnl = 0.0
        context.trade_log = []
        
        # Position tracking
        context.entry_price = None
        context.position_size = 0
        
    def handle_data(self, context: Context, data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals with bracket orders."""
        if data.is_empty():
            return []
            
        # Filter for our symbol
        symbol_data = data.filter(pl.col("symbol") == context.symbol)
        if symbol_data.is_empty() or len(symbol_data) < context.momentum_period:
            return []
            
        # Calculate momentum
        prices = symbol_data.select("close").to_series()
        current_price = prices[-1]
        
        if len(prices) < context.momentum_period:
            return []
            
        momentum = (current_price / prices[-context.momentum_period]) - 1.0
        
        signals = []
        current_position = context.portfolio.positions.get(context.symbol, 0)
        
        # Entry signal: Strong positive momentum and no current position
        if momentum > context.entry_threshold and current_position == 0:
            # Calculate position size
            max_investment = context.portfolio.cash * 0.2  # Use 20% of cash
            position_size = min(int(max_investment / current_price), context.max_position_size)
            
            if position_size > 0:
                # Calculate bracket order prices
                stop_loss_price = current_price * (1 - context.stop_loss_pct)
                take_profit_price = current_price * (1 + context.take_profit_pct)
                
                # Create bracket order with stop loss and take profit
                signals.append(Signal(
                    symbol=context.symbol,
                    quantity=position_size,
                    order_type="market",
                    # Bracket order parameters
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                    # Alternative: use trailing stop instead of fixed stop
                    # stop_trail_percent=context.trailing_stop_pct,
                ))
                
                context.entry_price = current_price
                context.position_size = position_size
                
        # Exit signal: Strong negative momentum (emergency exit)
        elif momentum < -context.entry_threshold and current_position > 0:
            signals.append(Signal(
                symbol=context.symbol,
                quantity=-current_position,
                order_type="market"
            ))
            
        return signals
    
    def on_trade(self, context: Context, trade_event: dict, portfolio) -> None:
        """Handle all trade executions including bracket orders."""
        context.trade_count += 1
        
        # Extract trade context
        trade_type = trade_event.get("trade_type", "regular")
        bracket_role = trade_event.get("bracket_role")
        order_type = trade_event.get("order_type", "unknown")
        symbol = trade_event["symbol"]
        quantity = trade_event["quantity"]
        price = trade_event["price"]
        value = trade_event["value"]
        
        # Enhanced trade logging
        trade_info = {
            "trade_number": context.trade_count,
            "timestamp": trade_event["timestamp"],
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": value,
            "trade_type": trade_type,
            "bracket_role": bracket_role,
            "order_type": order_type,
            "pnl": 0.0,  # Will be calculated below
        }
        
        # Handle different trade types with specific logic
        if trade_type == "bracket":
            # This is a bracket order execution (stop loss or take profit)
            if bracket_role == "take_profit":
                context.take_profit_trades += 1
                pnl = self._calculate_pnl(context, price, abs(quantity))
                context.total_pnl += pnl
                trade_info["pnl"] = pnl
                
                print(f"🎯 TAKE PROFIT EXECUTED!")
                print(f"   Symbol: {symbol}")
                print(f"   Quantity: {quantity}")
                print(f"   Price: ${price:.2f}")
                print(f"   P&L: ${pnl:.2f}")
                print(f"   Total P&L: ${context.total_pnl:.2f}")
                
                # Reset position tracking
                context.entry_price = None
                context.position_size = 0
                
            elif bracket_role == "stop_loss":
                context.stop_loss_trades += 1
                pnl = self._calculate_pnl(context, price, abs(quantity))
                context.total_pnl += pnl
                trade_info["pnl"] = pnl
                
                print(f"🛑 STOP LOSS TRIGGERED!")
                print(f"   Symbol: {symbol}")
                print(f"   Quantity: {quantity}")
                print(f"   Price: ${price:.2f}")
                print(f"   P&L: ${pnl:.2f}")
                print(f"   Total P&L: ${context.total_pnl:.2f}")
                
                # Implement adaptive risk management after stop loss
                self._handle_stop_loss_response(context, trade_event)
                
                # Reset position tracking
                context.entry_price = None
                context.position_size = 0
                
        elif trade_type == "stop_order":
            # Standalone stop order (not part of bracket)
            context.trailing_stop_trades += 1
            pnl = self._calculate_pnl(context, price, abs(quantity)) if quantity < 0 else 0.0
            if pnl != 0:
                context.total_pnl += pnl
                trade_info["pnl"] = pnl
            
            print(f"🔻 TRAILING STOP EXECUTED!")
            print(f"   Symbol: {symbol}")
            print(f"   Order Type: {order_type}")
            print(f"   Price: ${price:.2f}")
            if pnl != 0:
                print(f"   P&L: ${pnl:.2f}")
            
        else:
            # Regular entry or exit trade
            if quantity > 0:  # Entry trade
                context.entry_trades += 1
                print(f"📈 POSITION OPENED!")
                print(f"   Symbol: {symbol}")
                print(f"   Quantity: {quantity}")
                print(f"   Entry Price: ${price:.2f}")
                print(f"   Position Value: ${abs(value):.2f}")
                
            else:  # Exit trade (manual)
                pnl = self._calculate_pnl(context, price, abs(quantity))
                context.total_pnl += pnl
                trade_info["pnl"] = pnl
                
                print(f"📉 POSITION CLOSED (Manual Exit)!")
                print(f"   Symbol: {symbol}")
                print(f"   Quantity: {quantity}")
                print(f"   Exit Price: ${price:.2f}")
                print(f"   P&L: ${pnl:.2f}")
                
                # Reset position tracking
                context.entry_price = None
                context.position_size = 0
        
        # Log the trade
        context.trade_log.append(trade_info)
        
        # Print periodic summary
        if context.trade_count % 5 == 0:
            self._print_trade_summary(context)
    
    def _calculate_pnl(self, context: Context, exit_price: float, quantity: float) -> float:
        """Calculate P&L for a position exit."""
        if context.entry_price is None:
            return 0.0
        return (exit_price - context.entry_price) * quantity
    
    def _handle_stop_loss_response(self, context: Context, trade_event: dict) -> None:
        """Implement adaptive response to stop loss execution."""
        symbol = trade_event["symbol"]
        
        # Example adaptive risk management:
        # 1. Reduce position size for next trade
        # 2. Increase stop loss distance temporarily
        # 3. Wait before re-entering
        
        if not hasattr(context, 'adaptive_risk'):
            context.adaptive_risk = {}
            
        if symbol not in context.adaptive_risk:
            context.adaptive_risk[symbol] = {
                'stop_loss_count': 0,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0
            }
        
        # Increment stop loss count and adjust risk parameters
        context.adaptive_risk[symbol]['stop_loss_count'] += 1
        count = context.adaptive_risk[symbol]['stop_loss_count']
        
        # Reduce position size by 20% for each stop loss
        context.adaptive_risk[symbol]['position_size_multiplier'] *= 0.8
        
        # Increase stop loss distance by 10% for each stop loss
        context.adaptive_risk[symbol]['stop_loss_multiplier'] *= 1.1
        
        print(f"   🔧 ADAPTIVE RISK MANAGEMENT:")
        print(f"      Stop losses for {symbol}: {count}")
        print(f"      Position size multiplier: {context.adaptive_risk[symbol]['position_size_multiplier']:.2f}")
        print(f"      Stop loss multiplier: {context.adaptive_risk[symbol]['stop_loss_multiplier']:.2f}")
    
    def _print_trade_summary(self, context: Context) -> None:
        """Print comprehensive trade summary."""
        total_trades = context.trade_count
        if total_trades == 0:
            return
            
        print(f"\n📊 TRADE SUMMARY (after {total_trades} executions):")
        print(f"   Entry Trades: {context.entry_trades}")
        print(f"   Take Profit Trades: {context.take_profit_trades}")
        print(f"   Stop Loss Trades: {context.stop_loss_trades}")
        print(f"   Trailing Stop Trades: {context.trailing_stop_trades}")
        print(f"   Total P&L: ${context.total_pnl:.2f}")
        
        # Calculate win rate based on closed positions
        closed_positions = context.take_profit_trades + context.stop_loss_trades
        if closed_positions > 0:
            win_rate = context.take_profit_trades / closed_positions
            print(f"   Win Rate: {win_rate:.1%}")
            
        # Show recent trades
        print(f"   Recent Trades:")
        for trade in context.trade_log[-3:]:
            role = trade.get('bracket_role', 'regular')
            pnl_str = f"P&L: ${trade['pnl']:.2f}" if trade['pnl'] != 0 else ""
            print(f"     {trade['symbol']} {role} @ ${trade['price']:.2f} {pnl_str}")
        print()
    
    def analyze(self, context: Context, results: dict) -> None:
        """Final analysis of bracket order performance."""
        print("\n" + "="*60)
        print("FINAL BRACKET ORDER ANALYSIS")
        print("="*60)
        
        print(f"Total Executions: {context.trade_count}")
        print(f"Entry Trades: {context.entry_trades}")
        print(f"Take Profit Executions: {context.take_profit_trades}")
        print(f"Stop Loss Executions: {context.stop_loss_trades}")
        print(f"Trailing Stop Executions: {context.trailing_stop_trades}")
        print(f"Final P&L: ${context.total_pnl:.2f}")
        
        # Calculate effectiveness metrics
        total_exits = context.take_profit_trades + context.stop_loss_trades + context.trailing_stop_trades
        if total_exits > 0:
            tp_rate = context.take_profit_trades / total_exits
            sl_rate = context.stop_loss_trades / total_exits
            trail_rate = context.trailing_stop_trades / total_exits
            
            print(f"\nExit Distribution:")
            print(f"  Take Profit Rate: {tp_rate:.1%}")
            print(f"  Stop Loss Rate: {sl_rate:.1%}")
            print(f"  Trailing Stop Rate: {trail_rate:.1%}")
            
        # Show adaptive risk adjustments
        if hasattr(context, 'adaptive_risk'):
            print(f"\nAdaptive Risk Adjustments:")
            for symbol, risk_data in context.adaptive_risk.items():
                print(f"  {symbol}:")
                print(f"    Stop Losses: {risk_data['stop_loss_count']}")
                print(f"    Position Size Adj: {risk_data['position_size_multiplier']:.2f}")
                print(f"    Stop Loss Adj: {risk_data['stop_loss_multiplier']:.2f}")


if __name__ == "__main__":
    print("Bracket Order Strategy Example")
    print("="*50)
    print("This strategy demonstrates:")
    print("• Entry trades with automatic bracket orders")
    print("• Take profit execution tracking")
    print("• Stop loss execution tracking")
    print("• Adaptive risk management after stop losses")
    print("• Comprehensive trade type classification")
    print("\nTo run this strategy:")
    print()
    print("from vegas.engine import BacktestEngine")
    print("from vegas.examples.bracket_order_strategy import BracketOrderStrategy")
    print("from datetime import datetime")
    print()
    print("engine = BacktestEngine()")
    print("strategy = BracketOrderStrategy()")
    print("results = engine.run(")
    print("    start=datetime(2023, 1, 1),")
    print("    end=datetime(2023, 12, 31),")
    print("    strategy=strategy,")
    print("    initial_capital=100000")
    print(")")

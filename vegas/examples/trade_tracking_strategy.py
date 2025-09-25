"""
Example strategy demonstrating the use of the on_trade method for trade tracking.

This strategy shows how to use the on_trade callback to implement custom logic
when trades are executed, such as tracking trade performance, implementing
risk management, or logging trade details.
"""

from datetime import datetime
from typing import List
import polars as pl

from vegas.strategy import Strategy, Signal, Context


class TradeTrackingStrategy(Strategy):
    """Example strategy that tracks trade execution using the on_trade callback."""
    
    def initialize(self, context: Context) -> None:
        """Initialize strategy parameters and tracking variables."""
        context.symbol = "AAPL"
        context.sma_short = 10
        context.sma_long = 20
        
        # Trade tracking variables
        context.trade_count = 0
        context.total_commissions = 0.0
        context.total_volume = 0.0
        context.winning_trades = 0
        context.losing_trades = 0
        context.trade_log = []
        
        # Risk management
        context.max_position_size = 1000
        context.daily_loss_limit = -500.0
        context.daily_pnl = 0.0
        
    def handle_data(self, context: Context, data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals based on moving average crossover."""
        if data.is_empty():
            return []
            
        # Filter for our symbol
        symbol_data = data.filter(pl.col("symbol") == context.symbol)
        if symbol_data.is_empty() or len(symbol_data) < context.sma_long:
            return []
            
        # Calculate moving averages
        prices = symbol_data.select("close").to_series()
        sma_short = prices.tail(context.sma_short).mean()
        sma_long = prices.tail(context.sma_long).mean()
        current_price = prices[-1]
        
        signals = []
        
        # Get current position
        current_position = context.portfolio.positions.get(context.symbol, 0)
        
        # Entry signal: Short MA crosses above Long MA
        if sma_short > sma_long and current_position == 0:
            # Calculate position size based on available capital
            max_investment = context.portfolio.cash * 0.1  # Use 10% of cash
            position_size = min(int(max_investment / current_price), context.max_position_size)
            
            if position_size > 0:
                signals.append(Signal(
                    symbol=context.symbol,
                    quantity=position_size,
                    order_type="market"
                ))
                
        # Exit signal: Short MA crosses below Long MA or we have a position
        elif sma_short < sma_long and current_position > 0:
            signals.append(Signal(
                symbol=context.symbol,
                quantity=-current_position,  # Close entire position
                order_type="market"
            ))
            
        return signals
    
    def on_trade(self, context: Context, trade_event: dict, portfolio) -> None:
        """Handle trade execution events for tracking and risk management."""
        # Update trade tracking statistics
        context.trade_count += 1
        context.total_commissions += trade_event["commission"]
        context.total_volume += abs(trade_event["value"])
        
        # Determine trade category and log appropriately
        trade_type = trade_event.get("trade_type", "regular")
        bracket_role = trade_event.get("bracket_role")
        order_type = trade_event.get("order_type", "unknown")
        
        # Log trade details with enhanced context
        trade_info = {
            "trade_number": context.trade_count,
            "timestamp": trade_event["timestamp"],
            "symbol": trade_event["symbol"],
            "quantity": trade_event["quantity"],
            "price": trade_event["price"],
            "value": trade_event["value"],
            "commission": trade_event["commission"],
            "transaction_id": trade_event["transaction_id"],
            "trade_type": trade_type,
            "bracket_role": bracket_role,
            "order_type": order_type,
        }
        context.trade_log.append(trade_info)
        
        # Handle different types of trades
        if trade_type == "bracket":
            if bracket_role == "take_profit":
                print(f"✅ TAKE PROFIT: {trade_event['symbol']} "
                      f"Qty: {trade_event['quantity']} "
                      f"Price: ${trade_event['price']:.2f}")
                context.winning_trades += 1
            elif bracket_role == "stop_loss":
                print(f"🛑 STOP LOSS: {trade_event['symbol']} "
                      f"Qty: {trade_event['quantity']} "
                      f"Price: ${trade_event['price']:.2f}")
                context.losing_trades += 1
                
        elif trade_type == "stop_order":
            print(f"🔻 STOP ORDER: {trade_event['symbol']} "
                  f"Qty: {trade_event['quantity']} "
                  f"Price: ${trade_event['price']:.2f} "
                  f"Type: {order_type}")
                  
        else:  # Regular trade
            # Calculate P&L for this trade if it's a sell (position close/reduce)
            if trade_event["quantity"] < 0:  # Sell trade
                # This is a simplified P&L calculation
                # In practice, you might want to track cost basis more precisely
                if hasattr(context, 'entry_price') and context.entry_price:
                    pnl = (trade_event["price"] - context.entry_price) * abs(trade_event["quantity"])
                    context.daily_pnl += pnl
                    
                    if pnl > 0:
                        context.winning_trades += 1
                    else:
                        context.losing_trades += 1
                        
                    print(f"📈 TRADE CLOSED: {trade_event['symbol']} "
                          f"Qty: {trade_event['quantity']} "
                          f"Price: ${trade_event['price']:.2f} "
                          f"P&L: ${pnl:.2f}")
            else:  # Buy trade
                context.entry_price = trade_event["price"]
                print(f"📊 TRADE OPENED: {trade_event['symbol']} "
                      f"Qty: {trade_event['quantity']} "
                      f"Price: ${trade_event['price']:.2f}")
        
        # Risk management: Check daily loss limit
        if context.daily_pnl < context.daily_loss_limit:
            print(f"⚠️ Daily loss limit hit: ${context.daily_pnl:.2f}. "
                  f"Strategy should stop trading for the day.")
            # In a real implementation, you might set a flag to stop trading
            
        # Enhanced trade summary every 10 trades
        if context.trade_count % 10 == 0:
            win_rate = context.winning_trades / max(context.winning_trades + context.losing_trades, 1)
            
            # Count different trade types
            regular_trades = sum(1 for t in context.trade_log if t.get("trade_type") == "regular")
            bracket_trades = sum(1 for t in context.trade_log if t.get("trade_type") == "bracket")
            stop_trades = sum(1 for t in context.trade_log if t.get("trade_type") == "stop_order")
            
            print(f"\n📊 TRADE SUMMARY (after {context.trade_count} trades):")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Regular Trades: {regular_trades}")
            print(f"  Bracket Trades: {bracket_trades}")
            print(f"  Stop Orders: {stop_trades}")
            print(f"  Total Commissions: ${context.total_commissions:.2f}")
            print(f"  Total Volume: ${context.total_volume:.2f}")
            print(f"  Daily P&L: ${context.daily_pnl:.2f}\n")
    
    def before_trading_start(self, context: Context, data: pl.DataFrame) -> None:
        """Reset daily tracking variables at the start of each trading day."""
        context.daily_pnl = 0.0
        
    def analyze(self, context: Context, results: dict) -> None:
        """Perform final analysis and print trade statistics."""
        print("\n" + "="*50)
        print("FINAL TRADE ANALYSIS")
        print("="*50)
        print(f"Total Trades: {context.trade_count}")
        print(f"Winning Trades: {context.winning_trades}")
        print(f"Losing Trades: {context.losing_trades}")
        
        if context.winning_trades + context.losing_trades > 0:
            win_rate = context.winning_trades / (context.winning_trades + context.losing_trades)
            print(f"Win Rate: {win_rate:.1%}")
            
        print(f"Total Commissions Paid: ${context.total_commissions:.2f}")
        print(f"Total Volume Traded: ${context.total_volume:.2f}")
        
        # Calculate average trade size
        if context.trade_count > 0:
            avg_trade_size = context.total_volume / context.trade_count
            print(f"Average Trade Size: ${avg_trade_size:.2f}")
            
        print("\nLast 5 Trades:")
        for trade in context.trade_log[-5:]:
            print(f"  {trade['timestamp']}: {trade['symbol']} "
                  f"{trade['quantity']:+.0f} @ ${trade['price']:.2f}")


if __name__ == "__main__":
    # Example usage
    from vegas.engine import BacktestEngine
    from datetime import datetime
    
    print("This is an example strategy demonstrating the on_trade callback.")
    print("To run this strategy:")
    print()
    print("from vegas.engine import BacktestEngine")
    print("from vegas.examples.trade_tracking_strategy import TradeTrackingStrategy")
    print("from datetime import datetime")
    print()
    print("engine = BacktestEngine()")
    print("# engine.load_data(...)")  # Load your data
    print("strategy = TradeTrackingStrategy()")
    print("results = engine.run(")
    print("    start=datetime(2023, 1, 1),")
    print("    end=datetime(2023, 12, 31),")
    print("    strategy=strategy,")
    print("    initial_capital=100000")
    print(")")

# on_trade Callback Implementation

The `on_trade` method has been enhanced to provide a comprehensive callback mechanism for strategies to respond to trade executions.

## What Was Added

### 1. Enhanced Documentation
The `on_trade` method in the `Strategy` class now has comprehensive documentation explaining:
- When the method is called
- What parameters it receives
- The structure of the `trade_event` dictionary
- Use cases for implementing trade-related logic

### 2. Engine Integration
The `BacktestEngine` now calls the `on_trade` method immediately after each transaction is executed, providing:
- Real-time trade notifications
- Complete trade details
- Portfolio state after the trade
- Error handling to prevent strategy callback failures from breaking the backtest

### 3. Trade Event Structure
The `trade_event` dictionary passed to `on_trade` contains:
```python
{
    "timestamp": datetime,         # When the trade was executed
    "transaction_id": str,         # Unique identifier for this transaction
    "order_id": str,              # ID of the order that generated this trade
    "symbol": str,                # Asset symbol that was traded
    "quantity": float,            # Signed quantity (+ for buys, - for sells)
    "price": float,               # Execution price per share/unit
    "commission": float,          # Commission paid for this trade
    "value": float,               # Total trade value (quantity * price)
    # Enhanced context for edge cases
    "trade_type": str,            # "regular", "bracket", "stop_order"
    "bracket_role": str,          # "take_profit", "stop_loss", or None
    "parent_order_id": str,       # Parent order ID for bracket orders (or None)
    "oco_group_id": str,          # OCO group ID for linked orders (or None)
    "order_type": str,            # Original order type ("market", "limit", "stop", etc.)
}
```

## Usage Examples

### Basic Trade Tracking
```python
class MyStrategy(Strategy):
    def initialize(self, context):
        context.trade_count = 0
        context.total_commissions = 0.0
    
    def on_trade(self, context, trade_event, portfolio):
        context.trade_count += 1
        context.total_commissions += trade_event["commission"]
        
        print(f"Trade executed: {trade_event['symbol']} "
              f"{trade_event['quantity']:+.0f} @ ${trade_event['price']:.2f}")
```

### Risk Management
```python
def on_trade(self, context, trade_event, portfolio):
    # Track daily P&L
    if trade_event["quantity"] < 0:  # Sell/close position
        pnl = self.calculate_pnl(trade_event)
        context.daily_pnl += pnl
        
        # Check risk limits
        if context.daily_pnl < -1000:  # $1000 daily loss limit
            context.stop_trading = True
            print("Daily loss limit reached - stopping trading")
```

### Position Monitoring
```python
def on_trade(self, context, trade_event, portfolio):
    symbol = trade_event["symbol"]
    current_position = portfolio.positions.get(symbol, 0)
    
    if trade_event["quantity"] > 0:  # Opening/adding to position
        print(f"Opened/added to {symbol} position: {current_position}")
    else:  # Closing/reducing position
        print(f"Closed/reduced {symbol} position: {current_position}")
        if current_position == 0:
            print(f"Fully closed {symbol} position")
```

### Handling Edge Cases (Stop Loss & Take Profit)
```python
def on_trade(self, context, trade_event, portfolio):
    trade_type = trade_event.get("trade_type", "regular")
    bracket_role = trade_event.get("bracket_role")
    
    if trade_type == "bracket":
        if bracket_role == "take_profit":
            print(f"✅ TAKE PROFIT executed: {trade_event['symbol']} @ ${trade_event['price']:.2f}")
            context.winning_trades += 1
            # Handle successful profit-taking logic
            
        elif bracket_role == "stop_loss":
            print(f"🛑 STOP LOSS triggered: {trade_event['symbol']} @ ${trade_event['price']:.2f}")
            context.losing_trades += 1
            # Handle risk management after stop loss
            self._adjust_position_sizing(context, trade_event)
            
    elif trade_type == "stop_order":
        print(f"🔻 Stop order executed: {trade_event['order_type']}")
        # Handle standalone stop orders
        
    else:  # Regular trade
        # Handle regular entry/exit trades
        self._handle_regular_trade(context, trade_event, portfolio)

def _adjust_position_sizing(self, context, trade_event):
    """Reduce position size after stop loss to manage risk."""
    symbol = trade_event["symbol"]
    if hasattr(context, 'position_sizes'):
        # Reduce position size by 50% after a stop loss
        current_size = context.position_sizes.get(symbol, 1.0)
        context.position_sizes[symbol] = max(0.1, current_size * 0.5)
        print(f"Reduced position size for {symbol} to {context.position_sizes[symbol]:.1%}")
```

### Advanced Risk Management with OCO Orders
```python
def on_trade(self, context, trade_event, portfolio):
    oco_group = trade_event.get("oco_group_id")
    
    if oco_group:
        # This trade was part of an OCO group
        if not hasattr(context, 'oco_tracking'):
            context.oco_tracking = {}
            
        if oco_group not in context.oco_tracking:
            context.oco_tracking[oco_group] = []
            
        context.oco_tracking[oco_group].append(trade_event)
        
        # Check if this completes the OCO group
        group_trades = context.oco_tracking[oco_group]
        if len(group_trades) == 1:  # First trade in OCO executed
            other_orders_cancelled = True  # OCO logic cancels other orders
            print(f"OCO group {oco_group} executed: {trade_event['bracket_role']}")
```

## Implementation Details

### Timing
The `on_trade` callback is invoked:
1. **After** orders are executed by the broker
2. **Before** the portfolio is updated with transaction details
3. **Before** any other strategy hooks like `on_market_close`

### Error Handling
- If `on_trade` raises an exception, it's caught and logged as a warning
- The backtest continues normally - callback errors don't interrupt execution
- This ensures robust backtesting even with buggy callback implementations

### Performance Considerations
- The callback is called for each individual transaction
- For strategies that generate many trades, keep `on_trade` logic lightweight
- Consider aggregating data and performing heavy analysis in the `analyze` method

## Files Modified

1. **vegas/engine/engine.py**: Added callback invocation in the main backtest loop
2. **vegas/strategy/strategy.py**: Enhanced `on_trade` method documentation
3. **vegas/examples/trade_tracking_strategy.py**: Comprehensive example strategy
4. **vegas/tests/test_on_trade_callback.py**: Test suite for the functionality

## Benefits

1. **Real-time Trade Awareness**: Strategies can react immediately to trade executions
2. **Enhanced Risk Management**: Implement position limits, drawdown controls, etc.
3. **Trade Analytics**: Track performance metrics, win rates, commission costs
4. **Position Management**: Monitor position changes and implement complex logic
5. **Debugging**: Log detailed trade information for strategy development
6. **Edge Case Handling**: Comprehensive support for stop losses, take profits, and trailing stops
7. **Bracket Order Support**: Full visibility into OCO and bracket order executions
8. **Adaptive Risk Management**: React to stop losses with dynamic position sizing

## Edge Cases Covered

The enhanced `on_trade` callback captures **ALL** trade executions including:

### ✅ Bracket Orders
- **Take Profit Orders**: Automatically executed when profit targets are hit
- **Stop Loss Orders**: Triggered when stop levels are breached
- **Trailing Stops**: Dynamic stop orders that follow price movements
- **OCO (One-Cancels-Other)**: Linked orders where execution of one cancels others

### ✅ Standalone Stop Orders
- **Stop Market Orders**: Convert to market orders when triggered
- **Stop Limit Orders**: Convert to limit orders when triggered
- **Trailing Stop Orders**: Dynamic stops independent of bracket orders

### ✅ Regular Orders
- **Market Orders**: Immediate execution at current market price
- **Limit Orders**: Execution when price reaches specified level

The callback provides detailed context through the enhanced `trade_event` structure, enabling strategies to:
- Differentiate between regular trades and risk management executions
- Track the relationship between parent and child orders in bracket setups
- Implement adaptive risk management based on stop loss frequency
- Monitor the effectiveness of different exit strategies

The `on_trade` callback enables sophisticated trading strategies that need to monitor and react to their own trade executions in real-time, including comprehensive handling of all edge cases and advanced order types.

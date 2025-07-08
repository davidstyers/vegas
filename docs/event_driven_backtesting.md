# Event-Driven Backtesting in Vegas

## Overview

Vegas provides two backtesting modes to accommodate different strategy implementation approaches:

1. **Vectorized Mode (Default)**: Optimized for performance, processes all market data at once using efficient vector operations.
2. **Event-Driven Mode**: Processes market data sequentially through a series of events, allowing for more fine-grained control.

Event-driven backtesting is especially useful for strategies that:
- React to specific market events (e.g., market open/close)
- Need to maintain complex state between time periods
- Implement pre-market analysis
- Require more control over the execution timeline

## How to Use Event-Driven Mode

There are two ways to enable event-driven backtesting:

### 1. Automatic Detection

Simply implement any of the event handler methods in your strategy:
- `before_trading_start(context, data)`
- `on_market_open(context, data, portfolio)`
- `on_market_close(context, data, portfolio)`
- `on_bar(context, data)`
- `on_tick(context, data)`
- `on_trade(context, trade_event, portfolio)`

Vegas will automatically detect that your strategy requires event-driven execution.

### 2. Explicit Flag

You can also explicitly set the mode when creating your strategy:

```python
class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.is_event_driven = True  # Explicitly request event-driven mode
```

Or when running the backtest:

```python
engine.run(
    start=start_date,
    end=end_date,
    strategy=my_strategy,
    event_driven=True  # Explicitly request event-driven mode
)
```

## Available Events

Vegas provides several event types that you can implement in your strategies:

### `before_trading_start(context, data)`
- Called before market open each trading day (9:00 AM)
- Useful for computing signals, analyzing overnight developments, or preparing for the trading day
- Receives data for all symbols at the current time

### `on_market_open(context, data, portfolio)`
- Called at market open each trading day (9:30 AM)
- Useful for executing opening orders or processing signals from before_trading_start
- Receives data for all symbols and current portfolio state

### `on_market_close(context, data, portfolio)`
- Called at market close each trading day (4:00 PM)
- Useful for end-of-day position adjustments or reporting
- Receives data for all symbols and current portfolio state

### `on_bar(context, data)`
- Called when a new bar is received
- Useful for strategies based on OHLCV bars
- Receives data for all symbols for the current bar

### `on_tick(context, data)`
- Called when new tick data is received (for tick-level data)
- Useful for high-frequency strategies
- Receives tick data for all symbols

### `on_trade(context, trade_event, portfolio)`
- Called after a trade from this strategy is executed
- Useful for tracking trade performance or updating state after execution
- Receives details about the executed trade

### `handle_data(context, data)`
- Called for all events (required for signal generation)
- Should return a list of `Signal` objects to generate orders
- This is the main method for order generation in event-driven mode

## Signal Generation in Event-Driven Mode

Unlike vectorized mode, event-driven strategies generate signals through the `handle_data` method:

```python
def handle_data(self, context, data):
    signals = []
    
    # Example: Buy AAPL if a condition is met
    if should_buy_aapl(context, data):
        aapl_price = data[data['symbol'] == 'AAPL']['close'].iloc[0]
        signals.append(Signal(
            symbol='AAPL',
            action='buy',
            quantity=10,
            price=aapl_price
        ))
    
    return signals  # Return a list of Signal objects
```

## Performance Considerations

Event-driven backtesting is inherently slower than vectorized backtesting due to:
- Sequential processing vs. parallel operations
- More function calls and overhead
- Fine-grained data access patterns

However, Vegas uses Cython to optimize the event loop for the best possible performance. For strategies that require event-driven functionality, the performance penalty is minimized through:

1. Optimized Cython implementation of the event engine
2. Efficient event generation and scheduling
3. Minimal data copying and transformation
4. Fallback to Python when Cython is unavailable

## Example

Here's a simple event-driven strategy that demonstrates several event hooks:

```python
class EventDrivenMAStrategy(Strategy):
    def initialize(self, context):
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.short_window = 10  # Short MA window
        context.long_window = 30   # Long MA window
        context.signals = {}       # Store signals for each symbol
    
    def before_trading_start(self, context, data):
        # Calculate moving averages for each symbol
        for symbol in context.symbols:
            # Get symbol data and calculate signals
            symbol_data = data[data['symbol'] == symbol]
            if len(symbol_data) >= context.long_window:
                closes = symbol_data['close']
                short_ma = closes.rolling(window=context.short_window).mean()
                long_ma = closes.rolling(window=context.long_window).mean()
                
                # Check for crossover
                if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                    context.signals[symbol] = 'buy'
                elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                    context.signals[symbol] = 'sell'
    
    def on_market_open(self, context, data, portfolio):
        # Execute signals at market open
        # (actual orders are returned by handle_data)
        print(f"Market open: executing signals: {context.signals}")
    
    def handle_data(self, context, data):
        signals = []
        for symbol, signal_type in context.signals.items():
            if symbol in data['symbol'].values:
                price = data[data['symbol'] == symbol]['close'].iloc[0]
                
                if signal_type == 'buy':
                    signals.append(Signal(symbol=symbol, action='buy', quantity=10, price=price))
                elif signal_type == 'sell':
                    signals.append(Signal(symbol=symbol, action='sell', quantity=10, price=price))
        
        return signals
```

For a complete example, see the `event_driven_example.py` file in the examples directory.

## Debugging Event-Driven Strategies

When debugging event-driven strategies, consider:

1. Adding print statements to each event method to track execution flow
2. Setting the logger to DEBUG level for more detailed information
3. Creating a separate analyze method to review strategy behavior
4. Using explicit event types to control which events are generated

## Converting Between Modes

If you have an existing vectorized strategy and want to convert it to event-driven:

1. Implement the appropriate event methods based on your strategy's needs
2. Move signal generation logic from `generate_signals_vectorized` to event methods and `handle_data`
3. Return individual signals from `handle_data` instead of a signals DataFrame

Similarly, to convert an event-driven strategy to vectorized:

1. Move logic from event methods into `generate_signals_vectorized`
2. Return a complete signals DataFrame with all signals for the entire backtest period
3. Remove or simplify the event-specific methods 
# Event-Driven Backtesting in Vegas

## Overview

Vegas provides an event-driven backtesting approach that processes market data sequentially through a series of events, allowing for fine-grained control over strategy logic and execution.

Event-driven backtesting is particularly valuable for strategies that:
- React to specific market events (e.g., market open/close)
- Need to maintain complex state between time periods
- Implement pre-market analysis
- Require detailed control over the execution timeline

## Event System

The event-driven backtesting engine processes market data chronologically, generating the following event types:

### `before_trading_start`
- Called before market open each trading day (9:00 AM)
- Useful for computing signals, analyzing overnight developments, or preparing for the trading day
- Receives data for all symbols at the current time

### `on_market_open`
- Called at market open each trading day (9:30 AM)
- Useful for executing opening orders or processing signals from before_trading_start
- Receives data for all symbols and current portfolio state

### `on_market_close`
- Called at market close each trading day (4:00 PM)
- Useful for end-of-day position adjustments or reporting
- Receives data for all symbols and current portfolio state

### `on_bar`
- Called when a new bar is received
- Useful for strategies based on OHLCV bars
- Receives data for all symbols for the current bar

### `on_tick`
- Called when new tick data is received (for tick-level data)
- Useful for high-frequency strategies
- Receives tick data for all symbols

### `on_trade`
- Called after a trade from this strategy is executed
- Useful for tracking trade performance or updating state after execution
- Receives details about the executed trade

### `handle_data`
- Called for all events (required for signal generation)
- Should return a list of `Signal` objects to generate orders
- This is the main method for order generation

## Implementing Event-Driven Strategies

To create an event-driven strategy, subclass the `Strategy` class and implement the relevant event handlers:

```python
from vegas.strategy import Strategy, Signal

class MyEventDrivenStrategy(Strategy):
    def initialize(self, context):
        """Set up strategy parameters."""
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.short_window = 10
        context.long_window = 30
        context.signals = {}  # Store signals for each symbol

    def before_trading_start(self, context, data):
        """Calculate signals before market opens."""
        # Calculate moving averages for each symbol
        for symbol in context.symbols:
            # Get symbol data and calculate signals
            symbol_data = data[data['symbol'] == symbol]
            if len(symbol_data) >= context.long_window:
                closes = symbol_data.sort_values('timestamp')['close']
                short_ma = closes[-context.short_window:].mean()
                long_ma = closes[-context.long_window:].mean()

                # Check for crossover
                if short_ma > long_ma:
                    context.signals[symbol] = 'buy'
                elif short_ma < long_ma:
                    context.signals[symbol] = 'sell'

    def on_market_open(self, context, data, portfolio):
        """Execute signals at market open."""
        print(f"Market open: {data['timestamp'].iloc[0] if not data.empty else None}")
        # Actual orders will be generated in handle_data

    def handle_data(self, context, data):
        """Generate signals for the current event."""
        signals = []

        # Process each symbol with a signal
        for symbol, action in context.signals.items():
            symbol_data = data[data['symbol'] == symbol]

            # Skip if no data available
            if symbol_data.empty:
                continue

            price = symbol_data['close'].iloc[0]

            if action == 'buy':
                # Generate buy signal
                signals.append(Signal(
                    symbol=symbol,
                    action='buy',
                    quantity=10,  # Buy 10 shares
                    price=price
                ))
            elif action == 'sell':
                # Generate sell signal if we own shares
                if portfolio and symbol in portfolio.positions:
                    quantity = portfolio.positions[symbol]
                    signals.append(Signal(
                        symbol=symbol,
                        action='sell',
                        quantity=quantity,  # Sell all shares
                        price=price
                    ))

        # Clear signals after processing
        context.signals = {}

        return signals
```

## Running an Event-Driven Backtest

To run an event-driven backtest:

```python
from vegas.engine import BacktestEngine
from datetime import datetime

# Initialize engine
engine = BacktestEngine()

# Load data
engine.load_data("data/market_data.csv")

# Create strategy
strategy = MyEventDrivenStrategy()

# Run backtest
results = engine.run(
    start=datetime(2020, 1, 1),
    end=datetime(2020, 12, 31),
    strategy=strategy,
    initial_capital=100000.0
)

# Print results
print(f"Final Portfolio Value: ${results['equity_curve']['equity'].iloc[-1]:.2f}")
print(f"Total Return: {results['stats']['total_return_pct']:.2f}%")
```

## Performance Considerations

For optimal performance in event-driven backtesting, Vegas uses:

1. Cython implementation of the event engine when available
2. Efficient event generation and scheduling
3. Minimal data copying and transformation
4. Fallback to Python when Cython is unavailable

## Example

See `examples/event_driven_example.py` for a complete example of an event-driven strategy.

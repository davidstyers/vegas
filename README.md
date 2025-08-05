# Vegas: Event-Driven Backtesting Engine

A focused Python backtesting framework for event-driven trading strategies. Vegas provides a clean and efficient implementation for simulating market events and strategy interactions.

## Features

- **Event-Driven Architecture**: Sequential processing of market events for realistic simulation
- **High-Performance Data Processing**: Uses Polars for lightning-fast dataframe operations
- **Simple, Clean Design**: Focused codebase optimized for readability and extensibility
- **Comprehensive Analytics**: Performance metrics and visualization
- **DuckDB and Parquet**: Efficient data storage and querying
- **CLI Interface**: Easy to run backtests from the command line
- **Timezone Support**: Configure and normalize timestamps for international markets
- **Market Hours Handling**: Control regular vs. extended hours for accurate testing

## Installation

```bash
# Development installation
./install_dev.sh

# Regular installation
pip install .
```

## Quick Start

```python
from datetime import datetime
import polars as pl
from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Signal

class MovingAverageCrossover(Strategy):
    def initialize(self, context):
        context.short_window = 20
        context.long_window = 50
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        
    def before_trading_start(self, context, data):
        # Calculate signals before market opens
        for symbol in context.symbols:
            symbol_data = data.filter(pl.col('symbol') == symbol)
            if len(symbol_data) >= context.long_window:
                prices = symbol_data.sort('timestamp').get_column('close')
                short_ma = prices.tail(context.short_window).mean()
                long_ma = prices.tail(context.long_window).mean()
                
                if short_ma > long_ma:
                    # Bullish signal
                    context.signal = 'buy'
                elif short_ma < long_ma:
                    # Bearish signal
                    context.signal = 'sell'
                    
    def handle_data(self, context, data):
        signals = []
        for symbol in context.symbols:
            if hasattr(context, 'signal') and context.signal == 'buy':
                # Generate buy signal
                signals.append(Signal(
                    symbol=symbol,
                    action='buy',
                    quantity=10,
                    price=None  # Use market price
                ))
                
        return signals

# Create engine
engine = BacktestEngine()

# Load data
engine.load_data("data/example.csv")

# Create strategy
strategy = MovingAverageCrossover()

# Run backtest
results = engine.run(
    start=datetime(2020, 1, 1),
    end=datetime(2020, 12, 31),
    strategy=strategy,
    initial_capital=100000.0
)

# Print results
print(f"Final Portfolio Value: ${results['stats']['total_return'] + 100000:.2f}")
print(f"Total Return: {results['stats']['total_return_pct']:.2f}%")
print(f"Total Trades: {results['stats']['num_trades']}")
```

## Key Components

### 1. Engine
The `BacktestEngine` orchestrates the entire backtest process, managing data flow and event generation.

### 2. Strategy
Strategies respond to market events through callback methods:
- `initialize`: Set up strategy parameters
- `before_trading_start`: Run analysis before market opens
- `handle_data`: Process data at each event and generate signals
- `on_market_open`/`on_market_close`: Run special logic at market open/close
- `on_bar`/`on_tick`: Process bar or tick data events
- `on_trade`: Process executed trades
- `analyze`: Analyze backtest results

### 3. Portfolio
The `Portfolio` class tracks positions, cash, and performance metrics.

### 4. Data Layer
The `DataLayer` provides efficient access to market data using DuckDB and Parquet files.

## Advanced Usage

Check the `examples/` directory for more detailed examples and advanced features, including:

- Event-driven strategy implementations
- Different trading approaches
- Using the database system
- Customizing analysis

### Timezone Support

Vegas provides timezone handling to normalize timestamps across different markets:

```python
# Create engine with specific timezone
engine = BacktestEngine(timezone="US/Eastern")  

# Load data - timestamps will be automatically converted
engine.load_data("data/example.csv")
```

Using the CLI:

```bash
# Run backtest with New York timezone
vegas run my_strategy.py --timezone "US/Eastern" --start 2022-01-01 --end 2022-12-31

# Ingest data with Tokyo timezone
vegas ingest --file market_data.csv --timezone "Asia/Tokyo"
```

Supported timezones include:
- `UTC` (default)
- `US/Eastern` (New York)
- `US/Pacific` (Los Angeles)
- `Europe/London`
- `Europe/Paris`
- `Asia/Tokyo`
- `Australia/Sydney`
- And [many others](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

### Market Hours Handling

Vegas lets you control how regular and extended market hours are handled:

```python
# Set specific market hours
engine = BacktestEngine(timezone="US/Eastern")
engine.set_trading_hours("NASDAQ", open="09:30", close="16:00")

# Exclude extended hours data (pre-market and after-hours)
engine.ignore_extended_hours(True)
```

Using the CLI:

```bash
# Run with only regular market hours
vegas run my_strategy.py --regular-hours-only

# Customize market hours
vegas run my_strategy.py --market NYSE --market-open 09:30 --market-close 16:00
```

See [Market Hours Documentation](docs/market_hours.md) for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Getting Started with Vegas

This guide will help you get started with the Vegas vectorized backtesting engine.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vegas.git
cd vegas

# Install the package
pip install -e .
```

## Loading Data

Vegas can load data from CSV files (with optional Zstandard compression):

```python
from vegas.data import DataLayer

# Initialize the data layer
data_layer = DataLayer(data_dir="data")

# Load a single file
data_layer.load_data(file_path="data/sample_data.csv.zst")

# Or load multiple files from a directory
data_layer.load_multiple_files(directory="data/us-equities", file_pattern="*.ohlcv-1h.csv.zst")

# Get information about the loaded data
data_info = data_layer.get_data_info()
print(f"Loaded {data_info['row_count']} data points for {data_info['symbol_count']} symbols")
print(f"Date range: {data_info['start_date']} to {data_info['end_date']}")
```

## Creating a Vectorized Strategy

Creating a strategy involves subclassing the `Strategy` class and implementing the required methods:

```python
from vegas.strategy import Strategy, Context
import pandas as pd
import numpy as np

class SimpleMovingAverageStrategy(Strategy):
    def initialize(self, context):
        # Set strategy parameters
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.short_window = 10
        context.long_window = 30
        context.position_size = 100
    
    def generate_signals_vectorized(self, context, data):
        # Filter data for our symbols
        data = data[data['symbol'].isin(context.symbols)]
        
        if data.empty:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        signals = []
        
        # Process each symbol separately
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            # Calculate moving averages
            symbol_data['short_ma'] = symbol_data['close'].rolling(
                window=context.short_window).mean()
            symbol_data['long_ma'] = symbol_data['close'].rolling(
                window=context.long_window).mean()
            
            # Skip NaN values
            symbol_data = symbol_data.dropna()
            
            if len(symbol_data) <= 1:
                continue
            
            # Create shifted columns to detect crossovers
            symbol_data['prev_short_ma'] = symbol_data['short_ma'].shift(1)
            symbol_data['prev_long_ma'] = symbol_data['long_ma'].shift(1)
            
            # Skip the first row after shifting
            symbol_data = symbol_data.dropna()
            
            # Buy signal: short MA crosses above long MA
            symbol_data['buy_signal'] = (symbol_data['prev_short_ma'] <= symbol_data['prev_long_ma']) & \
                                       (symbol_data['short_ma'] > symbol_data['long_ma'])
            
            # Sell signal: short MA crosses below long MA
            symbol_data['sell_signal'] = (symbol_data['prev_short_ma'] >= symbol_data['prev_long_ma']) & \
                                        (symbol_data['short_ma'] < symbol_data['long_ma'])
            
            # Generate buy signals
            buy_signals = symbol_data[symbol_data['buy_signal']].apply(
                lambda row: {
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': context.position_size,
                    'price': None  # Market order
                }, axis=1
            ).tolist()
            
            # Generate sell signals
            sell_signals = symbol_data[symbol_data['sell_signal']].apply(
                lambda row: {
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': context.position_size,
                    'price': None  # Market order
                }, axis=1
            ).tolist()
            
            signals.extend(buy_signals)
            signals.extend(sell_signals)
        
        if not signals:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
            
        return pd.DataFrame(signals)
```

## Running a Backtest

Once you have a strategy, you can run a backtest:

```python
from vegas.engine import BacktestEngine
from my_strategy import SimpleMovingAverageStrategy
from datetime import datetime

# Initialize engine
engine = BacktestEngine()

# Load data
engine.load_data(file_path="data/sample_data.csv.zst")

# Create strategy
strategy = SimpleMovingAverageStrategy()

# Run backtest
results = engine.run(
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31),
    strategy=strategy,
    initial_capital=100000.0
)

# Print results
stats = results['stats']
print(f"Total Return: {stats['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
print(f"Number of Trades: {stats['num_trades']}")
```

## Command Line Usage

Vegas provides a command-line interface similar to Zipline for running backtests:

```bash
# Run a backtest with a strategy file
vegas run path/to/strategy.py --data-file data/sample_data.csv.zst --start 2022-01-01 --end 2022-12-31

# Save the equity curve plot
vegas run path/to/strategy.py --data-file data/sample_data.csv.zst --output equity_curve.png

# Save results to CSV
vegas run path/to/strategy.py --data-file data/sample_data.csv.zst --results-csv results.csv

# Run with data from a directory
vegas run path/to/strategy.py --data-dir data/us-equities --start 2022-01-01 --end 2022-12-31

# Get help
vegas run --help
```

### Creating a Strategy File

To run a strategy with the CLI, create a Python file with a class that inherits from `Strategy`:

```python
# my_strategy.py
from vegas.strategy import Strategy, Context

class MyStrategy(Strategy):
    def initialize(self, context):
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.ma_window = 20
        context.position_size = 100
    
    def generate_signals_vectorized(self, context, data):
        # Your strategy logic here
        # ...
        return signals_df
```

Then run it with the CLI:

```bash
vegas run my_strategy.py --data-file data/sample_data.csv.zst
```

## Vectorization Tips

For best performance with vectorized strategies:

1. Process all data at once using pandas/numpy operations
2. Use efficient data structures and avoid loops where possible
3. Leverage pandas' built-in methods like `rolling()`, `shift()`, etc.
4. Use boolean indexing for filtering data
5. Minimize redundant calculations by processing data by symbol

## Next Steps

- Check out the [example strategies](examples.md) for more ideas
- Learn about [advanced features](advanced_features.md)
- Read the [API reference](api_reference.md) for detailed documentation
- Explore the [CLI examples](../examples/README.md) for command-line usage 
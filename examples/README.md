# Vegas CLI Examples

This directory contains example strategies that can be run with the Vegas CLI.

## Running Examples

To run an example strategy, use the `vegas run` command followed by the path to the strategy file:

```bash
# Run a simple moving average strategy with data from a file
vegas run simple_ma_strategy.py --data-file ../data/sample_data.csv.zst --start 2020-01-01 --end 2021-01-01

# Run with data from a directory
vegas run simple_ma_strategy.py --data-dir ../data/us-equities --start 2020-01-01 --end 2021-01-01

# Save the equity curve plot
vegas run simple_ma_strategy.py --data-dir ../data/us-equities --output equity_curve.png

# Save results to CSV
vegas run simple_ma_strategy.py --data-dir ../data/us-equities --results-csv results.csv

# Enable verbose logging
vegas run simple_ma_strategy.py --data-dir ../data/us-equities -v
```

## Creating Your Own Strategies

To create your own strategy, create a Python file with a class that inherits from `vegas.strategy.Strategy`. The class must implement at least one of the following methods:

- `generate_signals_vectorized(self, context, data)`: For vectorized processing
- `handle_data(self, context, data)`: For event-driven processing

Example:

```python
from vegas.strategy import Strategy, Context

class MyStrategy(Strategy):
    def initialize(self, context):
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.lookback = 20
        context.position_size = 100
    
    def generate_signals_vectorized(self, context, data):
        # Your vectorized strategy logic here
        # ...
        return signals_df
```

## Available Examples

- `simple_ma_strategy.py`: A simple moving average crossover strategy 
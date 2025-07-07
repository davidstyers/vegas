# Vegas: Minimal Vectorized Backtesting Engine

A lightweight Python backtesting framework for quantitative trading strategies. Vegas leverages pandas and numpy for efficient vectorized operations.

## Features

- **Minimal design**: Small codebase, easy to understand and extend
- **Vectorized operations**: Fast backtesting using pandas and numpy
- **Comprehensive analytics**: Performance metrics and visualization with QuantStats
- **DuckDB and Parquet**: Efficient data storage and querying
- **CLI interface**: Easy to run backtests from the command line

## Installation

```bash
# Development installation
./install_dev.sh

# Regular installation
pip install .
```

## Quick Start

```python
import pandas as pd
from vegas.engine import BacktestEngine
from vegas.strategy import Strategy

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Create short and long moving averages
        signals['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['signal'][self.long_window:] = np.where(
            signals['short_ma'][self.long_window:] > signals['long_ma'][self.long_window:], 
            1.0, 0.0
        )
        
        # Generate positions
        signals['position'] = signals['signal'].diff()
        
        return signals

# Create engine
engine = BacktestEngine()

# Load data
engine.load_data("data/example.csv")

# Create strategy
strategy = MovingAverageCrossover(short_window=50, long_window=200)

# Run backtest
results = engine.run(
    start=pd.Timestamp("2020-01-01"),
    end=pd.Timestamp("2021-01-01"),
    strategy=strategy,
    initial_capital=100000
)

# Print results
print(results)
```

## Command Line Interface

Run backtests directly from the command line:

```bash
# Run a backtest with data file
vegas run examples/simple_ma_strategy.py --data-file data/example.csv --start 2020-01-01 --end 2021-01-01

# Run a backtest using already ingested data
vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01

# Ingest data into the database
vegas ingest --file data/example.csv

# Ingest OHLCV files into the database
vegas ingest-ohlcv --directory data

# Check database status
vegas db-status --detailed

# Run a SQL query on the database
vegas db-query --query "SELECT * FROM market_data LIMIT 10"
```

## Data Management

Vegas provides two ways to work with data:

### 1. In-memory (pandas)

Data is loaded directly into memory using pandas DataFrames. This is suitable for smaller datasets.

```python
# Load data from a file
engine.load_data(file_path="data/example.csv")

# Load data from multiple files in a directory
engine.load_data(directory="data/", file_pattern="*.csv")
```

### 2. DuckDB + Parquet (recommended for large datasets)

For larger datasets, Vegas offers a database system using DuckDB and Parquet:

```python
# Data is automatically ingested into the database when loading
engine.load_data(file_path="data/example.csv")

# Explicitly ingest data using CLI
vegas ingest --file data/example.csv

# Ingest OHLCV files
vegas ingest-ohlcv --directory data

# Run backtests with database data
vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01
```

Benefits of the DuckDB + Parquet approach:
- Efficient storage (compression)
- Fast querying
- SQL query support
- Reduced memory usage
- Partitioned storage for large datasets

### OHLCV Data Format

The system supports OHLCV (Open, High, Low, Close, Volume) files in the format:
- Filename pattern: `*.ohlcv-1h.csv.zst`
- Compressed with Zstandard
- CSV format with headers: ts_event, symbol, open, high, low, close, volume

Data is automatically partitioned by year, date, and symbol in the Parquet files stored in the `db` directory.

## Analytics and Reporting

Vegas provides comprehensive analytics through QuantStats:

```python
# Generate a performance report
from vegas.analytics import generate_tearsheet

generate_tearsheet(results['returns'], benchmark_returns=None, title="Strategy Performance")
```

## Creating Strategies

To create a strategy, subclass `Strategy`:

```python
from vegas.strategy import Strategy

class MyStrategy(Strategy):
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2
        
    def generate_signals(self, data):
        """Generate trading signals for each symbol.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        # Your strategy logic here
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        
        # Generate your signals...
        
        # Create positions
        signals['position'] = signals['signal'].diff()
        
        return signals
```

## Examples

See the `examples` directory for complete examples of strategies and backtests:

```bash
# Run an example
./run_example.sh

# Run the database example
./examples/database_example.sh

# Run the OHLCV ingestion example
./examples/ingest_ohlcv_example.sh
```

## Documentation

See the `docs` directory for more detailed documentation:

- [Getting Started](docs/getting_started.md)
- [Performance Optimization](docs/performance_optimization.md)
- [Benchmark Guide](docs/benchmark_guide.md)
- [Database System](docs/database_system.md)

## License

MIT License. See LICENSE file for details. 
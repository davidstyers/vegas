# Vegas: High-Performance Backtesting Engine

A modular, event-driven backtesting system for quantitative trading strategies, optimized for Python 3.13.5.

## Overview

Vegas is designed for backtesting trading strategies on large datasets with features to mitigate overfitting and prepare for future live trading capabilities.

## Architecture

- **Event-Driven Design**: Sequential processing of market data, signals, orders, and portfolio updates
- **Modular Components**:
  - Data Layer: Parquet/DuckDB-based historical market data management
  - Strategy Layer: Zipline-inspired API for trading strategy implementation
  - Broker Simulation: Order execution with slippage and commission models
  - Portfolio Layer: Tracks performance, risk metrics, and P&L
  - Backtest Engine: Coordinates all components in an event loop
  - Analytics: Performance statistics and visualizations

## Requirements

- Python 3.13.5
- Dependencies listed in requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from vegas.engine import BacktestEngine
from vegas.strategy import Strategy
from datetime import datetime

class MyStrategy(Strategy):
    def initialize(self, context):
        context.sma_window = 20

    def handle_data(self, context, data):
        # Strategy logic here
        signals = []
        # ... generate trading signals
        return signals

# Create and run backtest
engine = BacktestEngine()
engine.load_data("path/to/data.csv.zst")
results = engine.run(
    start=datetime(2015, 1, 1),
    end=datetime(2022, 12, 31),
    strategy=MyStrategy(),
    initial_capital=100000
)

# Analyze results
engine.get_results().plot_equity_curve()

# Generate QuantStats report
results.create_tearsheet(
    title="My Strategy Performance Report",
    benchmark_symbol="SPY",
    output_file="reports/my_strategy.html",
    output_format="html"
)

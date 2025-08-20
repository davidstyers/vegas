# Vegas Analytics - Hyperparameter Optimization

This module provides hyperparameter optimization utilities for the Vegas backtesting engine, built on top of Optuna.

## Features

- **Strategy Parameter Optimization**: Automatically find optimal parameters for trading strategies
- **Flexible Parameter Space**: Support for various parameter types (float, categorical, integer)
- **Resource Management**: Automatic cleanup of backtest resources
- **Optimization Summary**: Detailed reporting of optimization results
- **Easy Integration**: Simple API that works with any Vegas strategy

## Quick Start

```python
from datetime import datetime
from vegas.analytics import run_optimized_backtest, configure_logging

# Configure logging
configure_logging(verbose=True)

# Define your strategy factory
def create_strategy(**params):
    return MyStrategy(
        param1=params.get("param1", 0.5),
        param2=params.get("param2", 0.5),
    )

# Run optimized backtest
results = run_optimized_backtest(
    strategy_factory=create_strategy,
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 31),
    initial_capital=100000.0,
    optimize=True,
    n_trials=25,
)
```

## API Reference

### `configure_logging(verbose: bool = False)`

Configure logging for hyperparameter optimization.

**Parameters:**
- `verbose`: Whether to enable verbose logging

### `evaluate_strategy_params(strategy_factory, strategy_params, start_date, end_date, initial_capital=100000.0, logger=None)`

Evaluate strategy parameters by running a backtest.

**Parameters:**
- `strategy_factory`: Function that creates a strategy instance
- `strategy_params`: Parameters to pass to the strategy factory
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `initial_capital`: Initial capital for the backtest
- `logger`: Optional logger instance

**Returns:**
- Sharpe ratio from the backtest results

### `create_optuna_objective(strategy_factory, start_date, end_date, initial_capital=100000.0, logger=None, param_suggestions=None)`

Create an Optuna objective function for strategy optimization.

**Parameters:**
- `strategy_factory`: Function that creates a strategy instance
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `initial_capital`: Initial capital for the backtest
- `logger`: Optional logger instance
- `param_suggestions`: Dictionary mapping parameter names to Optuna suggestion functions

**Returns:**
- Optuna objective function

### `optimize_strategy(strategy_factory, start_date, end_date, initial_capital=100000.0, n_trials=25, study_name="vegas_optimization", direction="maximize", logger=None, param_suggestions=None, n_jobs=1)`

Optimize strategy parameters using Optuna.

**Parameters:**
- `strategy_factory`: Function that creates a strategy instance
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `initial_capital`: Initial capital for the backtest
- `n_trials`: Number of optimization trials
- `study_name`: Name for the Optuna study
- `direction`: Optimization direction ('maximize' or 'minimize')
- `logger`: Optional logger instance
- `param_suggestions`: Dictionary mapping parameter names to Optuna suggestion functions
- `n_jobs`: Number of parallel jobs for optimization

**Returns:**
- Tuple of (best_parameters, study)

### `run_optimized_backtest(strategy_factory, start_date, end_date, initial_capital=100000.0, optimize=False, n_trials=25, study_name="vegas_optimization", logger=None, param_suggestions=None, default_params=None)`

Run a backtest with optional hyperparameter optimization.

**Parameters:**
- `strategy_factory`: Function that creates a strategy instance
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `initial_capital`: Initial capital for the backtest
- `optimize`: Whether to perform hyperparameter optimization
- `n_trials`: Number of optimization trials (if optimize=True)
- `study_name`: Name for the Optuna study
- `logger`: Optional logger instance
- `param_suggestions`: Dictionary mapping parameter names to Optuna suggestion functions
- `default_params`: Default parameters to use if optimize=False

**Returns:**
- Backtest results dictionary

### `print_optimization_summary(study, logger=None)`

Print a summary of optimization results.

**Parameters:**
- `study`: Optuna study object
- `logger`: Optional logger instance

## Parameter Suggestions

You can define custom parameter suggestions for your optimization:

```python
import optuna

param_suggestions = {
    "momentum_factor": lambda trial: trial.suggest_float("momentum_factor", -0.1, 1.0),
    "volatility": lambda trial: trial.suggest_float("volatility", 0.01, 1.5),
    "trail": lambda trial: trial.suggest_float("trail", 0.05, 0.35),
    "entry_ema": lambda trial: trial.suggest_categorical("entry_ema", [144, 169, 288, 338, 610]),
    "window_length": lambda trial: trial.suggest_int("window_length", 10, 100),
}
```

## Examples

See `examples/hyperopt_example.py` for a complete working example.

### Generating QuantStats Reports

The engine now returns a `Results` object that has a built-in `create_tearsheet` method:

```python
# After running a backtest
results = engine.run(start, end, strategy, initial_capital)

# Generate a QuantStats report
results.create_tearsheet(
    title="My Strategy Performance Report",
    benchmark_symbol="SPY",
    output_file="reports/my_strategy_report.html",
    output_format="html"
)
```

Or use it in a strategy's `analyze` method:

```python
class MyStrategy(Strategy):
    def analyze(self, context, results):
        results.create_tearsheet(
            title="My Strategy Performance Report",
            benchmark_symbol="QQQ",
            output_file="reports/auto_generated_report.html",
            output_format="html"
        )
```

See `examples/strategy_with_report.py` for a complete example.

## Dependencies

- `optuna`: For hyperparameter optimization
- `tabulate`: For formatting optimization summaries

## Migration from vw.py

The hyperparameter optimization functions have been refactored from `dev/vw.py` into this library. To migrate existing code:

1. Replace direct Optuna usage with the library functions
2. Use `run_optimized_backtest()` instead of custom optimization loops
3. Define strategy factories instead of hardcoded strategy creation
4. Use the provided parameter suggestion system

Example migration:

```python
# Old way (in vw.py)
def run_backtest(optimize=False, n_trials=25):
    # ... custom optimization code ...
    pass

# New way
from vegas.analytics import run_optimized_backtest

def run_backtest(optimize=False, n_trials=25):
    return run_optimized_backtest(
        strategy_factory=create_strategy,
        start_date=start_date,
        end_date=end_date,
        optimize=optimize,
        n_trials=n_trials,
    )
```

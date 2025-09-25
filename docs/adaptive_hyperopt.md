# Adaptive Hyperparameter Optimization

The Vegas hyperopt module has been refactored to support adaptive optimization of not just strategy parameters, but also engine configuration parameters including frequency, data_type, and calendar settings.

## New Features

### 1. Engine Parameter Optimization

You can now optimize the following engine parameters alongside strategy parameters:

- **Frequency**: Data frequency or bar specification (e.g., "1h", "4h", "1d", "tick:1000")
- **Data Type**: Type of underlying data ("ohlcv", "tick", "tbbo")
- **Calendar**: Trading calendar ("NYSE", "24/7", "24/7_EST", etc.)

### 2. Flexible Configuration

The `OptimizationConfig` class now supports:

- **Fixed engine parameters**: Set specific values that won't be optimized
- **Optimizable engine parameters**: Define ranges for parameters to be optimized
- **Mixed configurations**: Some parameters fixed, others optimized

## Configuration Options

### Fixed Engine Parameters

Set engine parameters to fixed values that won't be optimized:

```python
config = OptimizationConfig(
    initial_capital=100_000.0,
    param_ranges={
        "window": {"type": "int", "low": 10, "high": 50},
    },
    # Fixed engine parameters
    frequency="1h",
    data_type="ohlcv",
    calendar="NYSE",
    objective_metric="sharpe_ratio",
    n_trials=100,
    study_name="fixed_engine_example",
    direction="maximize",
)
```

### Optimizable Engine Parameters

Define ranges for engine parameters to be optimized:

```python
config = OptimizationConfig(
    initial_capital=100_000.0,
    param_ranges={
        "window": {"type": "int", "low": 10, "high": 50},
    },
    # Engine parameter ranges for optimization
    frequency_ranges={
        "frequency": {"type": "categorical", "choices": ["1h", "4h", "1d"]}
    },
    data_type_ranges={
        "data_type": {"type": "categorical", "choices": ["ohlcv", "tick"]}
    },
    calendar_ranges={
        "calendar": {"type": "categorical", "choices": ["NYSE", "24/7"]}
    },
    objective_metric="sharpe_ratio",
    n_trials=100,
    study_name="adaptive_example",
    direction="maximize",
)
```

### Using Helper Functions

Use the provided helper functions to create engine parameter suggestions:

```python
from vegas.analytics.hyperopt import create_engine_suggestions

# Create suggestions for all engine parameters
engine_suggestions = create_engine_suggestions(
    frequencies=["1h", "4h", "1d"],
    data_types=["ohlcv", "tick"],
    calendars=["NYSE", "24/7", "24/7_EST"],
)

# Or create suggestions for specific parameters
from vegas.analytics.hyperopt import create_frequency_suggestions, create_calendar_suggestions

frequency_suggestions = create_frequency_suggestions(["1h", "4h", "1d"])
calendar_suggestions = create_calendar_suggestions(["NYSE", "24/7"])
```

## Usage Examples

### Basic Adaptive Optimization

```python
from vegas.analytics.hyperopt import optimize_strategy, create_engine_suggestions

# Create engine parameter suggestions
engine_suggestions = create_engine_suggestions(
    frequencies=["1h", "4h", "1d"],
    calendars=["NYSE", "24/7"],
)

# Run optimization
best_params, study = optimize_strategy(
    strategy_factory=my_strategy_factory,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 30),
    config=config,
    engine_param_suggestions=engine_suggestions,
)

# Separate strategy and engine parameters
strategy_params = {k: v for k, v in best_params.items() 
                  if k in config.param_ranges}
engine_params = {k: v for k, v in best_params.items() 
                if k in engine_suggestions}
```

### Mixed Configuration

```python
config = OptimizationConfig(
    initial_capital=100_000.0,
    param_ranges={
        "window": {"type": "int", "low": 10, "high": 50},
    },
    # Optimize frequency and calendar, fix data_type
    frequency_ranges={
        "frequency": {"type": "categorical", "choices": ["1h", "4h", "1d"]}
    },
    calendar_ranges={
        "calendar": {"type": "categorical", "choices": ["NYSE", "24/7"]}
    },
    data_type="ohlcv",  # Fixed
    objective_metric="sharpe_ratio",
    n_trials=100,
    study_name="mixed_config_example",
    direction="maximize",
)
```

## Supported Frequency Formats

The frequency parameter supports various formats:

- **Time-based**: "1h", "4h", "1d", "1w"
- **Tick-based**: "tick:1000", "tick:5000"
- **Volume-based**: "volume:10000", "volume:50000"
- **Dollar-based**: "dollar:100000", "dollar:500000"
- **Run bars**: "tick_run:1000", "volume_run:10000", "dollar_run:100000"
- **Imbalance bars**: "tick_imbalance:1000", "volume_imbalance:10000", "dollar_imbalance:100000"

## Supported Data Types

- **"ohlcv"**: Standard OHLCV bar data
- **"tick"**: Tick-level data (for tick bars, run bars, etc.)
- **"tbbo"**: Top-of-book bid/offer data

## Supported Calendars

- **"NYSE"**: NYSE trading hours (Mon-Fri, 9:30-16:00 EST)
- **"24/7"**: 24/7 trading (no filtering)
- **"24/7_EST"**: 24/7 trading in EST timezone
- **"24/7_CRYPTO"**: 24/7 trading in UTC timezone

## Best Practices

1. **Start Simple**: Begin with fixed engine parameters and optimize strategy parameters first
2. **Gradual Expansion**: Add engine parameter optimization once strategy parameters are stable
3. **Parameter Interaction**: Be aware that engine parameters can significantly affect strategy performance
4. **Computational Cost**: Engine parameter optimization increases the search space and computational cost
5. **Validation**: Always validate optimized parameters on out-of-sample data

## Performance Considerations

- Engine parameter optimization significantly increases the search space
- Consider reducing `n_trials` when optimizing engine parameters
- Use parallel processing (`n_jobs > 1`) when possible
- Start with smaller date ranges for initial optimization

## Migration from Legacy Hyperopt

The new interface is backward compatible. Existing code will continue to work, but you can now add engine parameter optimization by:

1. Adding `engine_param_suggestions` parameter to function calls
2. Using the new configuration options in `OptimizationConfig`
3. Using the helper functions for common parameter combinations

## Examples

See the following example files for detailed usage:

- `examples/hyperopt_adaptive_example.py`: Complete adaptive optimization example
- `examples/hyperopt_configuration_examples.py`: Various configuration approaches

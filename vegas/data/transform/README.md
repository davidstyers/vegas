# Vegas Data Transformation System

The Vegas data transformation system provides comprehensive frequency conversion and bar creation capabilities for both OHLCV and tick data. This implementation is based on Dr. Marcos López de Prado's methods from "Advances in Financial Machine Learning" and uses Polars for maximum efficiency.

## Overview

The system supports two main types of data transformations:

1. **OHLCV Resampling**: Convert time-based OHLCV data to different frequencies (e.g., 1h → 4h → 1d)
2. **Tick Bar Transformations**: Convert tick data into various bar types using Lopez de Prado's methods

## Quick Start

### Basic Usage in Backtesting

```python
from vegas.engine import BacktestEngine
from your_strategy import MyStrategy

engine = BacktestEngine()
strategy = MyStrategy()

# Time-based frequencies for OHLCV data
results = engine.run(
    start=start_date,
    end=end_date,
    strategy=strategy,
    frequency="4h",      # 4-hour bars
    data_type="ohlcv"
)

# Tick bars for tick data
results = engine.run(
    start=start_date,
    end=end_date,
    strategy=strategy,
    frequency="tick:1000",  # 1000-tick bars
    data_type="tick"
)
```

### Standalone Transformations

```python
from vegas.data.transform import TickBars, VolumeBars, OHLCVResampler
import polars as pl

# Transform tick data to tick bars
tick_transformer = TickBars(bar_size=1000)
tick_bars = tick_transformer.transform(tick_data)

# Resample OHLCV data to daily bars
ohlcv_resampler = OHLCVResampler(target_frequency="1d")
daily_bars = ohlcv_resampler.transform(hourly_data)
```

## Supported Frequencies

### OHLCV Data (Time-Based)

| Frequency | Description |
|-----------|-------------|
| `1min`    | 1-minute bars |
| `5min`    | 5-minute bars |
| `15min`   | 15-minute bars |
| `1h`      | 1-hour bars (default) |
| `4h`      | 4-hour bars |
| `1d`      | Daily bars |
| `1w`      | Weekly bars |

### Tick Data (Bar-Based)

| Bar Type | Format | Description |
|----------|--------|-------------|
| Tick Bars | `tick:N` | N ticks per bar |
| Volume Bars | `volume:N` | N shares per bar |
| Dollar Bars | `dollar:N` | $N dollar volume per bar |
| Tick Imbalance Bars | `tick_imbalance:N` | Based on tick flow imbalance |
| Volume Imbalance Bars | `volume_imbalance:N` | Based on volume flow imbalance |
| Dollar Imbalance Bars | `dollar_imbalance:N` | Based on dollar flow imbalance |
| Tick Run Bars | `tick_run:N` | Based on tick run sequences |
| Volume Run Bars | `volume_run:N` | Based on volume run sequences |
| Dollar Run Bars | `dollar_run:N` | Based on dollar run sequences |

## Detailed Usage

### OHLCV Resampling

The `OHLCVResampler` converts OHLCV data to different time frequencies:

```python
from vegas.data.transform import OHLCVResampler

# Resample to 4-hour bars
resampler = OHLCVResampler(target_frequency="4h")
resampled_data = resampler.transform(ohlcv_df)

# Multi-frequency resampling
from vegas.data.transform import MultiFrequencyResampler

multi_resampler = MultiFrequencyResampler(["1h", "4h", "1d"])
all_frequencies = multi_resampler.transform(ohlcv_df)
```

**Required Columns**: `timestamp`, `open`, `high`, `low`, `close`, `volume`

### Tick Bars

Standard tick bars group ticks by count:

```python
from vegas.data.transform import TickBars

transformer = TickBars(bar_size=1000)  # 1000 ticks per bar
tick_bars = transformer.transform(tick_data)
```

**Required Columns**: `timestamp`, `price`
**Optional Columns**: `volume` (defaults to tick count if missing)

### Volume Bars

Volume bars group by cumulative share volume:

```python
from vegas.data.transform import VolumeBars

transformer = VolumeBars(bar_size=10000)  # 10,000 shares per bar
volume_bars = transformer.transform(tick_data)
```

**Required Columns**: `timestamp`, `price`, `volume`

### Dollar Bars

Dollar bars group by cumulative dollar volume:

```python
from vegas.data.transform import DollarBars

transformer = DollarBars(bar_size=100000)  # $100,000 per bar
dollar_bars = transformer.transform(tick_data)
```

**Required Columns**: `timestamp`, `price`, `volume`

### Imbalance Bars

Imbalance bars create bars based on order flow imbalance (Lopez de Prado Chapter 2):

```python
from vegas.data.transform import TickImbalanceBars, VolumeImbalanceBars, DollarImbalanceBars

# Tick imbalance bars
tick_imbalance = TickImbalanceBars(
    bar_size=1000,
    expected_imbalance=0.0  # Expected imbalance ratio
)
bars = tick_imbalance.transform(tick_data)

# Volume imbalance bars
volume_imbalance = VolumeImbalanceBars(bar_size=10000)
bars = volume_imbalance.transform(tick_data)

# Dollar imbalance bars  
dollar_imbalance = DollarImbalanceBars(bar_size=100000)
bars = dollar_imbalance.transform(tick_data)
```

**Algorithm**:
1. Calculate tick rule (buy/sell classification) based on price changes
2. Compute cumulative imbalance (tick rule * volume for volume bars)
3. Create new bar when absolute imbalance exceeds threshold

### Run Bars

Run bars create bars based on sequences of consecutive buy or sell orders:

```python
from vegas.data.transform import TickRunBars, VolumeRunBars, DollarRunBars

# Tick run bars
tick_run = TickRunBars(bar_size=1000)
bars = tick_run.transform(tick_data)

# Volume run bars
volume_run = VolumeRunBars(bar_size=10000)
bars = volume_run.transform(tick_data)

# Dollar run bars
dollar_run = DollarRunBars(bar_size=100000)
bars = dollar_run.transform(tick_data)
```

**Algorithm**:
1. Calculate tick rule (buy/sell classification)
2. Identify runs of consecutive buy or sell ticks
3. Create bars when run meets size threshold

## Data Format Requirements

### Input Data Schemas

**OHLCV Data**:
```python
schema = {
    "timestamp": pl.Datetime,
    "symbol": pl.Utf8,      # Optional for single symbol
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
```

**Tick Data**:
```python
schema = {
    "timestamp": pl.Datetime,
    "symbol": pl.Utf8,      # Optional for single symbol
    "price": pl.Float64,
    "volume": pl.Float64,   # Optional for some transformations
}
```

### Output Schema

All transformations produce OHLCV-format output:

```python
output_schema = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
```

## Frequency Manager

The `FrequencyManager` provides utilities for parsing and validating frequencies:

```python
from vegas.data.transform import FrequencyManager
from vegas.data.transform.base import DataType

# Parse frequency string
freq_spec = FrequencyManager.parse_frequency("tick:1000", DataType.TICK)

# Validate frequency for data type
is_valid = FrequencyManager.validate_frequency_for_data_type(freq_spec)

# Get available frequencies
ohlcv_freqs = FrequencyManager.get_available_frequencies(DataType.OHLCV)
tick_freqs = FrequencyManager.get_available_frequencies(DataType.TICK)

# Get transformer class
transformer_class = FrequencyManager.get_transformer_class(freq_spec)
```

## Integration with Vegas Engine

The transformation system is fully integrated with the Vegas backtesting engine:

### Engine Integration

```python
# The engine automatically:
# 1. Parses frequency specification
# 2. Validates compatibility with data type
# 3. Loads appropriate raw data
# 4. Applies transformations
# 5. Provides transformed data to strategy

results = engine.run(
    start=start_date,
    end=end_date,
    strategy=strategy,
    frequency="volume:5000",  # Automatically parsed and applied
    data_type="tick"
)
```

### CLI Integration

```bash
# OHLCV frequencies
vegas run strategy.py --start 2023-01-01 --end 2023-12-31 --frequency 4h

# Tick bar frequencies
vegas run strategy.py --start 2023-01-01 --end 2023-01-31 --frequency tick:1000 --data-type tick
```

## Performance Considerations

### Optimization Tips

1. **OHLCV Resampling**: Very efficient using Polars `group_by_dynamic`
2. **Tick Bars**: Efficient for large datasets using vectorized operations
3. **Imbalance/Run Bars**: More computationally intensive due to sequential logic
4. **Memory Usage**: Tick transformations may require pandas conversion for iterative algorithms

### Recommended Bar Sizes

| Asset Class | Tick Bars | Volume Bars | Dollar Bars |
|-------------|-----------|-------------|-------------|
| Large Cap Stocks | 1000-5000 | 10K-50K | $100K-500K |
| Small Cap Stocks | 500-2000 | 5K-20K | $50K-200K |
| ETFs | 1000-3000 | 10K-30K | $100K-300K |
| Futures | 500-2000 | 5K-25K | $50K-250K |

## Error Handling

The transformation system includes comprehensive error handling:

```python
try:
    transformer = TickBars(bar_size=1000)
    result = transformer.transform(data)
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Transformation error: {e}")
    # System returns original data if transformation fails
```

## Examples

See `examples/frequency_examples.py` for comprehensive usage examples including:

- OHLCV frequency resampling
- All tick bar transformation types
- Standalone transformation usage
- Integration with backtesting

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. [MLFinPy Documentation](https://mlfinpy.readthedocs.io/) - Reference pandas implementation
3. [Polars Documentation](https://pola-rs.github.io/polars/) - For time series operations

## Migration from Other Systems

### From MLFinPy

```python
# MLFinPy (pandas)
from mlfinpy.data_structures import get_tick_bars
tick_bars = get_tick_bars(data, threshold=1000)

# Vegas (polars)
from vegas.data.transform import TickBars
transformer = TickBars(bar_size=1000)
tick_bars = transformer.transform(data)
```

### From Zipline

```python
# Zipline
data = bundles.load('quandl', assets=['SPY'], start='2020-01-01')

# Vegas
engine = BacktestEngine()
results = engine.run(start, end, strategy, frequency="1d", data_type="ohlcv")
```

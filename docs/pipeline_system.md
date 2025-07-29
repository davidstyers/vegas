# Pipeline System for Vegas

## Overview

The Vegas pipeline system is a powerful vectorized data processing engine that allows you to define and compute cross-sectional computations on market data. Inspired by Zipline's pipeline, it precomputes data before the trading day starts, so you can make better decisions without slowing down your strategy execution.

## Core Components

The pipeline system consists of these key components:

1. **Pipeline**: A container for computations to be executed together
2. **Terms**: Base building blocks like Factors, Filters, and Classifiers
3. **Factors**: Components that compute numerical values (like moving averages)
4. **Filters**: Components that compute boolean values (for screening assets)
5. **PipelineEngine**: Engine that executes pipeline computations

## Getting Started

### Creating a Pipeline

```python
from vegas.pipeline import Pipeline, SimpleMovingAverage, Returns

# Create a Pipeline with factors
my_pipeline = Pipeline(
    columns={
        'sma_10': SimpleMovingAverage(inputs=['close'], window_length=10),
        'daily_returns': Returns(window_length=2),  # 1-day returns
        'monthly_returns': Returns(window_length=21)  # Approx. 1-month returns
    }
)
```

### Adding a Screen

You can filter the universe of assets with a screen:

```python
from vegas.pipeline import StaticAssets

# Create a Pipeline with a screen
tech_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
tech_filter = StaticAssets(tech_stocks)

my_pipeline = Pipeline(
    columns={
        'sma_10': SimpleMovingAverage(inputs=['close'], window_length=10),
        'daily_returns': Returns(window_length=2)
    },
    screen=tech_filter  # Only compute for tech stocks
)
```

### Attaching a Pipeline to your Strategy

```python
class MyStrategy(Strategy):
    def initialize(self, context):
        # Create pipeline
        pipe = Pipeline(
            columns={
                'sma_10': SimpleMovingAverage(inputs=['close'], window_length=10),
                'daily_returns': Returns(window_length=2),
            },
            screen=StaticAssets(['AAPL', 'MSFT', 'GOOG'])
        )
        
        # Attach the pipeline to the engine
        context._engine.attach_pipeline(pipe, 'my_pipeline')
    
    def before_trading_start(self, context, data):
        # Get the pipeline results for today
        results = context._engine.pipeline_output('my_pipeline')
        
        # Use results to set up trading signals
        stocks_with_positive_returns = results[results['daily_returns'] > 0]
        context.candidates = stocks_with_positive_returns.index.get_level_values('symbol').tolist()
    
    def handle_data(self, context, data):
        signals = []
        
        # Generate buy signals for candidate stocks
        for symbol in context.candidates:
            if symbol in data['symbol'].values:
                signals.append(Signal(symbol=symbol, action="BUY", quantity=10))
        
        return signals
```

## Creating Custom Factors

You can create custom factors by subclassing `CustomFactor`:

```python
from vegas.pipeline import CustomFactor
import numpy as np

class Momentum(CustomFactor):
    """
    Custom factor that calculates momentum based on returns over different periods.
    """
    inputs = ['close']
    window_length = 126  # 6 months of trading days
    
    def compute(self, today, assets, out, closes):
        # Calculate returns over different periods
        monthly_returns = (closes[-1] / closes[-21] - 1)  # 1-month return
        quarterly_returns = (closes[-1] / closes[-63] - 1)  # 3-month return
        biannual_returns = (closes[-1] / closes[-126] - 1)  # 6-month return
        
        # Weight the returns (more weight to recent periods)
        out[:] = 0.5 * monthly_returns + 0.3 * quarterly_returns + 0.2 * biannual_returns
```

## Built-in Factors

Vegas comes with several built-in factors:

- `Returns`: Computes percentage change in price over a period
- `SimpleMovingAverage`: Computes a simple moving average
- `ExponentialWeightedMovingAverage`: Computes an exponential moving average
- `VWAP`: Computes volume-weighted average price
- `StandardDeviation`: Computes standard deviation of inputs
- `ZScore`: Normalizes values across assets
- `Rank`: Computes sorted ranks of assets
- `Percentile`: Computes percentile ranks of assets

## Built-in Filters

Vegas includes these built-in filters:

- `StaticAssets`: Filters for a static list of assets
- `BinaryCompare`: Filters based on comparisons (>, <, ==, etc.)
- `NotNaN`: Filters out assets with missing data
- `All`: Requires all input filters to be True
- `Any`: Requires at least one input filter to be True
- `AtLeastN`: Requires at least N input filters to be True
- `NotMissing`: Filters assets with non-missing data

## Pipeline Results

Pipeline results are returned as pandas DataFrames with a MultiIndex of (date, symbol):

```python
>>> results = engine.pipeline_output('my_pipeline')
>>> results.head()
                      sma_10  daily_returns  monthly_returns
date       symbol                                          
2020-01-10 AAPL     105.234         0.015           0.056
           MSFT     102.456         0.012           0.043
           GOOG     101.123         0.008           0.032
```

## Advanced Usage

### Factor Operations

Factors support arithmetic operations:

```python
# Create a factor that represents the ratio of price to 50-day SMA
sma_50 = SimpleMovingAverage(inputs=['close'], window_length=50)
price_to_sma = Returns(window_length=1, inputs=['close']) / sma_50
```

### Filter Combinations

Filters support boolean operations:

```python
# Create a filter for liquid stocks with positive returns
liquid_stocks = SimpleMovingAverage(inputs=['volume'], window_length=30) > 100000
positive_returns = Returns(window_length=5) > 0
good_stocks = liquid_stocks & positive_returns
```

### Factor Methods

Factors provide helpful methods like:

```python
# Get the top 10 stocks by momentum
momentum = Momentum()
top_10_momentum = momentum.top(10)

# Get stocks with positive z-scores
zscore_momentum = momentum.zscore()
positive_momentum = zscore_momentum > 0
```

## Example Pipeline Strategy

See the `examples/pipeline_example.py` file for a complete example of a strategy using the pipeline system. 
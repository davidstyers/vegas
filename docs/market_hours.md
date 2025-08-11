# Market Hours Handling in Vegas

Vegas provides tools to handle both regular and extended market hours during backtesting. This documentation explains how to use these features.

## Why Market Hours Matter

Different markets operate on different schedules, and many markets have extended trading sessions before and after regular hours. For accurate backtesting, it's important to be able to:

1. **Filter data** - Test strategies that only trade during regular market hours
2. **Identify extended hours** - Analyze pre-market or after-hours behavior separately
3. **Market-specific rules** - Apply different trading rules based on market hours
4. **Accurate simulation** - Ensure backtests reflect real-world trading conditions

## Configuration Options

### Setting Trading Hours

You can configure what Vegas considers "regular market hours" for your backtest:

```python
# In Python code
engine = BacktestEngine(timezone="US/Eastern")
engine.set_trading_hours("NASDAQ", open="09:30", close="16:00")
```

### Filtering Extended Hours Data

You can choose to include or exclude extended hours data in your backtest:

```python
# To ignore pre-market and after-hours data
engine.ignore_extended_hours(True)

# To include all data (default behavior)
engine.ignore_extended_hours(False)
```

## Using from the Command Line

The CLI provides options to configure market hours:

```bash
# Run backtest with only regular market hours data
vegas run strategy.py --regular-hours-only

# Customize market hours
vegas run strategy.py --market NYSE --market-open 09:30 --market-close 16:00

# Run with extended hours included (default)
vegas run strategy.py
```

## How It Works

When regular hours filtering is enabled:

1. Data points outside regular market hours are filtered out
2. The `before_trading_start` strategy callback uses the first data point from regular market hours
3. Portfolio values are only updated using data from regular market hours

## Common Market Hours

| Market | Regular Hours (Local Time) |
|--------|----------------------------|
| NYSE/NASDAQ (US) | 09:30 - 16:00 |
| LSE (London) | 08:00 - 16:30 |
| TSE (Tokyo) | 09:00 - 15:00 |
| SSE (Shanghai) | 09:30 - 15:00 |
| FSX (Frankfurt) | 09:00 - 17:30 |
| ASX (Australia) | 10:00 - 16:00 |

## Best Practices

1. **Timezone consistency** - Ensure your market hours configuration matches your data timezone
2. **Date handling** - Remember that filtering extended hours may result in missing days if you only have extended hours data for some dates
3. **Testing both** - Consider running backtests both with and without extended hours to understand their impact on your strategy
4. **Understand your data** - Make sure your dataset properly marks or timestamps extended hours sessions

## Example Strategy

```python
class RegularHoursStrategy(Strategy):
    def initialize(self, context):
        # Setup for regular hours trading
        context.symbols = ["AAPL", "MSFT", "GOOG"]

    def before_trading_start(self, context, data):
        # This will be called at the start of regular trading hours
        # (e.g., 9:30 AM for US markets)
        self.log.info(f"Market open: {data['timestamp'].iloc[0]}")

    def handle_data(self, context, data):
        # This will only be called for data points during regular hours
        # if ignore_extended_hours is enabled
        pass
```

## Debugging Tips

- Check logs for how many data points are filtered out
- Verify the market hours settings match your expectations
- Examine timestamps in your strategy callbacks to confirm filtering
- Compare results with and without extended hours to understand the impact

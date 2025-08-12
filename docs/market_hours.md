# Calendar-based Market Hours in Vegas

Vegas uses a unified calendar system to regulate market days and sessions (including holidays). This documentation explains how to use calendars to filter market data during backtesting.

## Why Market Hours Matter

Different markets operate on different schedules, and many markets have extended trading sessions before and after regular hours. For accurate backtesting, it's important to be able to:

1. **Filter data** - Test strategies that only trade during regular market hours
2. **Identify extended hours** - Analyze pre-market or after-hours behavior separately
3. **Market-specific rules** - Apply different trading rules based on market hours
4. **Accurate simulation** - Ensure backtests reflect real-world trading conditions

## Configuration Options

### Selecting a Calendar

Choose a trading calendar by name. Built-ins include `"NYSE"` (weekdays 09:30–16:00, no holidays modeled yet) and `"24/7"` (no filtering):

```python
from vegas.engine import BacktestEngine

engine = BacktestEngine()
engine.set_calendar("NYSE")  # engine timezone follows the calendar
```

The engine passes the selected calendar to the data portal which filters timestamps accordingly.

## Using from the Command Line

The CLI provides an option to select the calendar:

```bash
# Run backtest with NYSE calendar
vegas run strategy.py --calendar NYSE

# Run with 24/7 calendar (default)
vegas run strategy.py --calendar "24/7"
```

## How It Works

When a calendar is selected:

1. Data points outside the calendar session are filtered out
2. The `before_trading_start` strategy callback is invoked per trading day
3. Portfolio values are updated using calendar-valid data points

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

1. **Timezone consistency** - Ensure your calendar choice matches your data timezone
2. **Holiday coverage** - Extend calendars with holiday lists for production usage
3. **Compare calendars** - Try `"NYSE"` vs `"24/7"` to understand session impacts
4. **Understand your data** - Ensure timestamps reflect local session times

## Example Strategy

```python
class RegularHoursStrategy(Strategy):
    def initialize(self, context):
        # Setup for regular hours trading
        context.symbols = ["AAPL", "MSFT", "GOOG"]

    def before_trading_start(self, context, data):
        # Called once per trading day according to the calendar
        pass

    def handle_data(self, context, data_portal):
        # Called for calendar-valid timestamps only
        return []
```

## Debugging Tips

- Check logs for how many data points are filtered out
- Verify the market hours settings match your expectations
- Examine timestamps in your strategy callbacks to confirm filtering
- Compare results with and without extended hours to understand the impact

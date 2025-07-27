# Timezone Handling in Vegas

When backtesting strategies across international markets or dealing with data from different timezones, proper timezone handling is essential for accurate results. Vegas provides comprehensive timezone support that allows you to normalize timestamps for consistent strategy execution.

## Why Timezone Handling Matters

Consider the following scenarios where timezone handling is crucial:

1. **Trading across international markets**: If your strategy trades in both US and European markets, comparing timestamps without timezone normalization can lead to execution errors.

2. **Time-based signals**: Strategies that rely on specific times of day (market open/close, lunch hours) need accurate timezone information.

3. **Data consistency**: Data from various sources may have different timezone information or none at all.

4. **Overnight positions**: Correctly accounting for overnight positions and end-of-day calculations requires proper timezone handling.

## Using Timezone Support

### In Python Code

```python
from datetime import datetime
from vegas.engine import BacktestEngine

# Create engine with specific timezone
engine = BacktestEngine(timezone="US/Eastern")

# Load data - timestamps will be automatically converted to US/Eastern
engine.load_data("data/example.csv")

# Run backtest with dates in the specified timezone
results = engine.run(
    start=datetime(2022, 1, 1),
    end=datetime(2022, 12, 31),
    strategy=my_strategy,
    initial_capital=100000.0
)
```

### From Command Line

```bash
# Run with New York timezone
vegas run my_strategy.py --timezone "US/Eastern" --start 2022-01-01 --end 2022-12-31

# Run with Tokyo timezone
vegas run my_strategy.py --timezone "Asia/Tokyo" --start 2022-01-01 --end 2022-12-31

# Run with London timezone
vegas run my_strategy.py --timezone "Europe/London" --start 2022-01-01 --end 2022-12-31
```

## How Timezone Handling Works

Vegas handles timezones in the following way:

1. **Input Data**: When loading data, timestamps without timezone information are assumed to be in UTC.

2. **Conversion**: All timestamps are converted to the specified timezone during the data loading process.

3. **Strategy Execution**: All timestamps passed to strategy callbacks are in the configured timezone.

4. **Results**: All timestamps in the results (equity curve, transactions, etc.) use the configured timezone.

## Accessing Timestamp Information in Strategies

Within your strategy callbacks, you can access timestamp information in the configured timezone:

```python
def handle_data(self, context, data):
    # Get the current timestamp from data
    timestamp = data['timestamp'].iloc[0]
    
    # Check current hour in the configured timezone
    current_hour = timestamp.hour
    
    # Check day of week
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    
    # Format timestamp for logging
    formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
    self.logger.info(f"Processing data at {formatted_time}")
```

## Common Timezone Identifiers

Vegas uses the IANA timezone database via pytz. Some common timezone identifiers:

| Region | Identifier | Description |
|--------|------------|-------------|
| UTC | "UTC" | Coordinated Universal Time (default) |
| United States | "US/Eastern" | Eastern Time (New York) |
| United States | "US/Central" | Central Time (Chicago) |
| United States | "US/Pacific" | Pacific Time (Los Angeles) |
| Europe | "Europe/London" | UK Time |
| Europe | "Europe/Paris" | Central European Time |
| Asia | "Asia/Tokyo" | Japan Time |
| Asia | "Asia/Shanghai" | China Time |
| Australia | "Australia/Sydney" | Sydney Time |

For a full list of available timezones, see the [IANA Time Zone Database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

## Best Practices

1. **Be consistent**: Choose a single timezone for your backtesting workflow and stick with it.

2. **Use UTC for storage**: When storing market data, use UTC timezone as a standard practice.

3. **Match market timezone**: For strategies focused on a single market, consider using that market's native timezone (e.g., US/Eastern for US equities).

4. **Handle DST changes**: Be aware that Daylight Saving Time changes can affect time-based strategies. Vegas handles these transitions automatically.

5. **Document timezone usage**: Always document which timezone your strategy is designed to work with.

## Example

See `examples/timezone_example.py` for a complete demonstration of timezone handling in Vegas. 
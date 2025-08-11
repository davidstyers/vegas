# Timezone Handling in Vegas

## Overview

Vegas provides robust timezone handling capabilities to ensure consistent processing of timestamp data across different markets and data sources. This document explains how timestamps are handled throughout the system.

## Key Components

### 1. Database-level Timezone Conversion

Timestamps are stored in the database in UTC format for consistency. When querying data from the database, Vegas can convert timestamps to the target timezone directly in the database query, which offers several advantages:

- **Performance**: Converting at the database level is generally faster than post-query conversion
- **Consistency**: All timestamp data is handled uniformly with clear timezone information
- **Accuracy**: Avoids potential issues with daylight saving time transitions

### 2. Data Layer Timezone Configuration

The `DataLayer` class is configured with a timezone parameter that defines the target timezone for all data processing:

```python
# Initialize with Eastern time
data_layer = DataLayer(timezone="America/New_York")

# Or use UTC (default)
data_layer = DataLayer(timezone="UTC")
```

This timezone setting is used throughout the system:

1. When querying data from the database
2. When loading data from files
3. When converting timestamps in memory

### 3. Engine-level Configuration

The `BacktestEngine` class inherits the timezone setting and passes it to the data layer:

```python
# Create engine with timezone
engine = BacktestEngine(timezone="Asia/Tokyo")
```

## Implementation Details

### Database Queries

When querying data, the timezone conversion happens directly in the SQL query:

```sql
SELECT
    CAST(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' AS TIMESTAMP) AS timestamp,
    symbol, open, high, low, close, volume, source
FROM market_data
```

This approach ensures that:

1. All timestamp data is stored consistently in UTC
2. The timezone conversion happens efficiently at the database level
3. Applications receive the data in their preferred timezone

### In-memory Conversion

For in-memory data or when the database is not available, timestamps are converted using Polars:

```python
df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", time_zone=self.timezone)))
```

## Best Practices

1. **Always specify a timezone** when initializing the system to ensure consistent behavior
2. **Use IANA timezone identifiers** (e.g., "America/New_York" instead of "EST")
3. **Consider your data sources' timezones** when analyzing data and generating signals

## Timezone and Trading Hours

When specifying market hours for regular trading hours filtering, be aware of the timezone context:

```python
# New York market hours in Eastern Time
engine.set_trading_hours(market_name="NYSE", open_time="09:30", close_time="16:00")
```

The trading hours are interpreted in the context of the configured timezone.

## Testing Timezone Handling

You can test the timezone handling using the `test_timezone_db_conversion.py` example script, which:

1. Creates sample data with timestamps
2. Tests conversion with multiple timezones
3. Compares database-level conversion vs. post-query conversion
4. Verifies timestamp accuracy and performance

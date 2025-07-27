# Timezone Handling Implementation

This document describes how timezone support was implemented in the Vegas backtesting engine.

## Components Modified

1. **DataLayer**
   - Added `timezone` parameter to the constructor
   - Added timezone validation using pytz
   - Implemented `_convert_timestamp_timezone` method for converting dataframes
   - Modified data loading methods to handle timezone conversion
   - Updated methods that work with timestamps to be timezone-aware

2. **BacktestEngine**
   - Added `timezone` parameter to the constructor
   - Passes timezone to DataLayer
   - Stores timezone as an instance variable for future use

3. **CLI Interface**
   - Added `--timezone` parameter to common argument parser
   - Modified command handlers to pass timezone to engine
   - Added logging for timezone settings
   - Updated results display to include timezone info

4. **Dependencies**
   - Added `pytz` to dependencies in both `pyproject.toml` and `setup.py`

5. **Documentation**
   - Added timezone handling guide
   - Updated README with timezone features
   - Added timezone example to demonstrate usage

## Implementation Details

### DataLayer Timezone Handling

The core of the timezone handling is implemented in the `DataLayer` class:

```python
def _convert_timestamp_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe timestamps to the configured timezone."""
    if df is None or df.empty or 'timestamp' not in df.columns:
        return df
        
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Make sure timestamps are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # If timestamps don't have timezone info, assume UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
    # Convert to target timezone
    if self.timezone != 'UTC' or df['timestamp'].dt.tz.zone != self.timezone:
        df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)
        
    return df
```

This method handles:
1. Ensuring timestamps are datetime objects
2. Adding timezone info if missing (assuming UTC)
3. Converting to the target timezone

### Key Workflow

The timezone handling works as follows:

1. User specifies a timezone when creating the engine or via CLI
2. DataLayer validates the timezone and falls back to UTC if invalid
3. When data is loaded, all timestamps are converted to the specified timezone
4. All subsequent operations (strategy callbacks, portfolio updates) work with the normalized timestamps
5. All outputs (equity curve, transactions) use the consistent timezone

### Backward Compatibility

For backward compatibility:
- UTC is the default timezone if none is specified
- Existing code continues to work without modification
- Timestamps without timezone information are assumed to be in UTC

### Known Limitations

1. **Database Integration**: Some timezone handling may need further integration with the DuckDB layer.
2. **Serialization**: If results are serialized/deserialized, timezone information might be lost.
3. **Performance**: Converting large datasets between timezones may have performance implications.

## Future Improvements

1. **Timezone Conversion Optimization**: Optimize the conversion process for very large datasets
2. **Cross-Market Trading**: Enhanced support for trading across markets in different timezones
3. **Market Calendar Integration**: Integrate timezone handling with market calendars
4. **Config File Support**: Allow timezone settings to be specified in configuration files
5. **Timezone Metadata**: Add timezone metadata to saved backtest results 
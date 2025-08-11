# Migration Guide: Pandas to Polars

Vegas has been migrated from pandas to [polars](https://www.pola.rs/) to improve performance and reduce memory usage. This guide will help you adapt your code to the new polars-based implementation.

## Why Polars?

Polars offers several advantages over pandas:

1. **Performance**: Polars is significantly faster than pandas for most operations
2. **Memory Efficiency**: Polars uses less memory through more efficient data structures
3. **API Design**: Polars has a cleaner, more consistent API
4. **Parallelism**: Polars automatically parallelizes operations where possible
5. **Arrow Integration**: Polars is built on Arrow, providing excellent integration with the PyData ecosystem

## Key API Changes

### DataFrame Creation

**Pandas:**
```python
import pandas as pd
df = pd.DataFrame({
    'symbol': ['AAPL', 'MSFT', 'GOOG'],
    'price': [150.0, 250.0, 2000.0]
})
```

**Polars:**
```python
import polars as pl
df = pl.DataFrame({
    'symbol': ['AAPL', 'MSFT', 'GOOG'],
    'price': [150.0, 250.0, 2000.0]
})
```

### Empty DataFrame Creation

**Pandas:**
```python
df = pd.DataFrame(columns=['symbol', 'price'])
```

**Polars:**
```python
df = pl.DataFrame(schema={'symbol': pl.Utf8, 'price': pl.Float64})
```

### Reading Files

**Pandas:**
```python
df = pd.read_csv("data.csv")
```

**Polars:**
```python
df = pl.read_csv("data.csv")
```

### Boolean Filtering

**Pandas:**
```python
filtered = df[df['price'] > 200.0]
```

**Polars:**
```python
filtered = df.filter(pl.col('price') > 200.0)
```

### Selecting Columns

**Pandas:**
```python
symbols = df['symbol']
```

**Polars:**
```python
symbols = df.select('symbol').to_series()
# or
symbols = df.get_column('symbol')
```

### Adding Columns

**Pandas:**
```python
df['returns'] = df['price'] * 0.05
```

**Polars:**
```python
df = df.with_columns(pl.col('price') * 0.05).alias('returns')
```

### Sorting

**Pandas:**
```python
sorted_df = df.sort_values('price')
```

**Polars:**
```python
sorted_df = df.sort('price')
```

### Grouping

**Pandas:**
```python
grouped = df.groupby('symbol').agg({'price': 'mean'})
```

**Polars:**
```python
grouped = df.group_by('symbol').agg(pl.col('price').mean())
```

### Row Access

**Pandas:**
```python
first_row = df.iloc[0]
```

**Polars:**
```python
first_row = df.row(0)
# or
first_row = df.head(1)
```

### Checking Empty DataFrame

**Pandas:**
```python
if df.empty:
    print("DataFrame is empty")
```

**Polars:**
```python
if df.is_empty():
    print("DataFrame is empty")
```

## Strategy Pattern Changes

The main changes in Vegas strategies involve:

1. Use `filter()` instead of boolean indexing
2. Use `sort()` instead of `sort_values()`
3. Use `is_empty()` instead of checking the `empty` property
4. Use `get_column()` or `select().to_series()` instead of accessing columns with `df['column']`
5. Use `with_columns()` instead of direct assignment for adding columns

## Example Strategy Before and After

**Before (Pandas):**
```python
def handle_data(self, context, data):
    for symbol in context.symbols:
        symbol_data = data[data['symbol'] == symbol]
        if not symbol_data.empty:
            symbol_data['sma'] = symbol_data['close'].rolling(20).mean()
            symbol_data = symbol_data.sort_values('timestamp')
            latest_price = symbol_data['close'].iloc[-1]
            latest_sma = symbol_data['sma'].iloc[-1]

            if latest_price > latest_sma:
                # Generate buy signal
```

**After (Polars):**
```python
def handle_data(self, context, data):
    for symbol in context.symbols:
        symbol_data = data.filter(pl.col('symbol') == symbol)
        if not symbol_data.is_empty():
            symbol_data = symbol_data.sort('timestamp')
            prices = symbol_data.get_column('close')
            sma = prices.rolling_mean(window_size=20)
            symbol_data = symbol_data.with_columns(sma.alias('sma'))

            latest_price = prices.last()
            latest_sma = sma.last()

            if latest_price > latest_sma:
                # Generate buy signal
```

## Need Help?

If you encounter issues during migration or have questions about how to translate a specific pandas operation to polars, please open an issue on GitHub.

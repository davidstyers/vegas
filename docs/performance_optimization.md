# Performance Optimization Guide

This document outlines the performance optimization techniques implemented in the Vegas backtesting engine to minimize runtime and improve efficiency when working with large datasets.

## Overview

Backtesting financial strategies often involves processing large volumes of market data across multiple securities and time periods. The Vegas engine incorporates various optimization techniques to significantly reduce runtime while maintaining accuracy.

## Optimization Techniques

### 1. Database Optimizations

#### DuckDB Integration
Vegas uses DuckDB, an analytical database optimized for OLAP workloads:
- In-memory processing for maximum speed
- Vectorized query execution
- Parallel query processing
- Columnar storage format

#### Query Optimization
- **Prepared Statements**: Pre-compiled SQL queries that execute faster and reduce parsing overhead
- **SQL Optimization**: Carefully crafted queries to minimize data movement and processing
- **Indexes and Views**: Strategic use of materialized views for frequently accessed data patterns
- **Memory Management**: Configurable memory limits to prevent out-of-memory issues

```python
# Example: Using prepared statements
self.get_bars_query = self.db_conn.prepare("""
    SELECT timestamp, symbol, open, high, low, close, volume
    FROM market_data
    WHERE symbol = ?
      AND timestamp >= ?
      AND timestamp <= ?
    ORDER BY timestamp ASC
""")

# Later, execute with parameters
result = self.get_bars_query.execute([symbol, start, end]).fetchdf()
```

### 2. Data Storage and Access

#### Partitioned Data Storage
- Data is partitioned by year and symbol for efficient access
- Enables faster filtering and reduces I/O when accessing specific subsets of data

#### Batch Operations
- **Batch Data Retrieval**: Fetch data for multiple symbols in a single query
- **Bulk Loading**: Process and store multiple data files in coordinated operations

```python
def get_bars_batch(self, symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    """Retrieve OHLCV bars for multiple symbols in a single query."""
    result = self.get_bars_batch_query.execute([symbols, start, end]).fetchdf()

    # Split the result by symbol
    symbol_dfs = {}
    for symbol in symbols:
        symbol_data = result[result['symbol'] == symbol]
        if not symbol_data.empty:
            symbol_dfs[symbol] = symbol_data.reset_index(drop=True)

    return symbol_dfs
```

#### Caching System
- **Query Result Caching**: Store results of expensive queries for reuse
- **Universe Caching**: Cache available symbols for each date
- **Date-Symbol Mapping**: Pre-build maps of which symbols are available on which dates

```python
@cache_result
def get_universe(self, date: datetime) -> List[str]:
    """Get the list of available symbols for a given date."""
    date_key = pd.Timestamp(date).date()
    if date_key in self._date_symbol_map:
        return sorted(list(self._date_symbol_map[date_key]))

    # Otherwise query the database
    result = self.get_universe_query.execute([date.date()]).fetchdf()
    symbols = result['symbol'].tolist()

    # Update the cache
    self._date_symbol_map[date_key] = set(symbols)

    return symbols
```

### 3. Parallel Processing

#### Multi-Processing for File Loading
- Distributes file loading across multiple CPU cores
- Each worker processes a chunk of files independently
- Results are merged after parallel processing

```python
def load_multiple_files(self, directory: str = None, parallel: bool = True, num_workers: int = None):
    """Load multiple files in parallel."""
    if parallel and len(file_paths) > 1:
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(file_paths))

        # Divide files among workers
        chunks = np.array_split(file_paths, num_workers)

        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(
                    self._process_file_chunk, list(chunk), temp_dirs[i]))

            # Wait for all tasks to complete
            for future in futures:
                future.result()
```

#### Multi-Threading for Data Queries
- Use thread pools for I/O bound operations
- Enables concurrent data access without the overhead of multiple processes

### 4. Memory Optimization

#### Efficient Data Structures
- Using appropriate data structures for specific access patterns
- Dictionary-based lookups for fast symbol and date retrieval

#### Memory Limiting
- Configurable memory limits for DuckDB
- Prevents out-of-memory errors with large datasets

```python
def __init__(self, data_dir: str = "data", mem_limit_gb: float = None):
    """Initialize with optional memory limits."""
    if mem_limit_gb:
        # Convert GB to bytes
        mem_limit_bytes = int(mem_limit_gb * 1024 * 1024 * 1024)
        config = {"memory_limit": f"{mem_limit_bytes}B"}

    self.db_conn = duckdb.connect(":memory:", config=config)
```

#### Data Chunking
- Process large datasets in manageable chunks
- Reduces peak memory usage during operations

### 5. Algorithmic Improvements

#### Batch Strategy Evaluation
- Run multiple strategies on the same data without reloading
- Share data loading and preprocessing costs across strategies

```python
def run_multiple_strategies(self, strategies: List[Strategy], start: datetime, end: datetime):
    """Run multiple backtests with different strategies on the same data."""
    # Gather all required symbols from all strategies
    all_symbols = set()
    for strategy in strategies:
        if hasattr(strategy, 'required_symbols'):
            all_symbols.update(strategy.required_symbols)

    # Preload data once for all strategies
    self._preload_data(start, end, list(all_symbols))

    # Run each strategy
    results = {}
    for strategy in strategies:
        strategy_result = self.run(
            start=start,
            end=end,
            strategy=strategy,
            preload_data=False  # Already preloaded
        )
        results[strategy.__class__.__name__] = strategy_result
```

#### Data Preloading
- Preload all required data at the start of a backtest
- Avoids repeated database queries during the backtest loop

```python
def _preload_data(self, start: datetime, end: datetime, symbols: List[str] = None):
    """Preload data for the backtest period."""
    # Preload universe for each date in the range
    date_range = pd.date_range(start.date(), end.date())
    for date in date_range:
        self.get_universe(date)

    # Preload data for each symbol in batches
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        self.get_bars_batch(batch, start, end)
```

#### Vectorized Operations
- Leverage pandas and numpy for vectorized calculations
- Much faster than iterative approaches for numerical operations

```python
# Instead of looping through each row
for i in range(len(prices)):
    if prices[i] < 0:
        raise ValueError("Negative price found")

# Use vectorized operations
if (prices < 0).any():
    raise ValueError("Negative price found")
```

### 6. Additional Techniques

#### Profiling and Monitoring
- Built-in timing and logging to identify bottlenecks
- Memory usage tracking to detect inefficiencies

#### PyArrow Integration
- Use PyArrow for efficient data interchange between pandas and Parquet
- Reduces memory overhead when loading and saving data

```python
# Using PyArrow for efficient data processing
table = pa.Table.from_pandas(data)
pq.write_table(table, file_path, compression='snappy')
```

#### Compression Optimization
- Use Snappy compression for Parquet files
- Balances compression ratio with decompression speed

## Benchmarking

To measure the performance improvements from these optimizations, use the provided benchmarking tool:

```bash
# Run benchmark with optimizations
python benchmark_backtester.py --data-dir data/us-equities

# Run benchmark without optimizations for comparison
python benchmark_backtester.py --data-dir data/us-equities --no-optimize
```

## Advanced Optimization Tips

### Hardware Considerations
- Use SSDs for data storage when possible
- Ensure sufficient RAM for your dataset size
- Multi-core CPUs benefit the parallel processing features

### Customizing for Your Workload
- Adjust batch sizes based on your specific dataset and hardware
- Tune the number of worker processes to match your CPU core count
- Set memory limits appropriate for your system

```python
# Customize engine for your hardware
engine = BacktestEngine(
    mem_limit_gb=16,  # Adjust based on available RAM
    num_workers=12     # Adjust based on CPU cores
)
```

### Additional Optimizations for Very Large Datasets
- Consider memory-mapped files for datasets larger than RAM
- Implement custom index structures for specialized lookups
- Use distributed computing frameworks like Dask for multi-node processing

## Conclusion

These optimization techniques allow Vegas to efficiently process large financial datasets while maintaining accuracy. The system's modular design allows for selecting the appropriate optimization techniques based on your specific backtesting needs and hardware capabilities.

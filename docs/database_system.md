# DuckDB and Parquet Database System

Vegas includes a powerful database system using DuckDB and Parquet for efficient data storage and querying. This approach offers significant benefits for backtesting with large datasets.

## Overview

The database system consists of:

1. **DuckDB**: An in-process SQL OLAP database management system
2. **Parquet Files**: A columnar storage file format designed for efficient data storage and retrieval
3. **Integration with Vegas**: Seamless integration with the existing DataLayer and CLI

## Benefits

- **Efficient Storage**: Parquet's columnar format with compression reduces disk space usage
- **Fast Querying**: DuckDB provides efficient SQL queries directly on Parquet files
- **Reduced Memory Usage**: Only load the data you need, when you need it
- **SQL Interface**: Use SQL to query and analyze your market data
- **Partitioning**: Data is partitioned by year and symbol for efficient access
- **Compatibility**: Fallback to in-memory pandas if database not available

## Architecture

The database system consists of two main components:

1. **ParquetManager**: Handles writing and reading Parquet files
2. **DatabaseManager**: Provides a high-level interface to the DuckDB database

### Database Schema

The Vegas database includes the following tables and views:

- `market_data`: View of all market data from Parquet files
- `symbols`: Table with available symbols
- `data_sources`: Table tracking ingested data sources

## Using the Database System

### From the CLI

```bash
# Ingest data into the database
vegas ingest --file data/example.csv

# Check database status
vegas db-status --detailed --show-symbols

# Run a SQL query on the database
vegas db-query --query "SELECT * FROM market_data LIMIT 10"

# Save query results to a file
vegas db-query --query "SELECT * FROM market_data WHERE symbol = 'AAPL'" --output apple_data.csv

# Run a backtest using database data (no need to specify a data file)
vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01
```

### From Python

```python
from vegas.data import DataLayer
from vegas.engine import BacktestEngine

# Initialize data layer with database
data_layer = DataLayer(data_dir="data")

# Check database status
info = data_layer.get_data_info()
print(f"Database contains {info['row_count']} rows for {info['symbol_count']} symbols")

# Load data into the database
data_layer.load_data("data/example.csv")  # Automatically ingests into database

# Get data from the database
data = data_layer.get_data_for_backtest(
    start=pd.Timestamp("2020-01-01"),
    end=pd.Timestamp("2021-01-01"),
    symbols=["AAPL", "MSFT"]
)

# Direct SQL query
if data_layer.db_manager:
    result = data_layer.db_manager.query_to_df(
        "SELECT * FROM market_data WHERE symbol = 'AAPL' LIMIT 10"
    )
    print(result)

# Run backtest using database data
engine = BacktestEngine()
engine.data_layer = data_layer  # Use initialized data layer

# Create strategy and run backtest (no need to load data again)
strategy = MyStrategy()
results = engine.run(
    start=pd.Timestamp("2020-01-01"),
    end=pd.Timestamp("2021-01-01"),
    strategy=strategy,
    initial_capital=100000
)
```

## Advanced Usage

### Custom SQL Queries

You can run any SQL query supported by DuckDB:

```bash
vegas db-query --query "
SELECT
    symbol,
    DATE_TRUNC('month', timestamp) as month,
    AVG(close) as avg_price,
    MAX(high) as max_high,
    MIN(low) as min_low,
    SUM(volume) as total_volume
FROM market_data
WHERE symbol IN ('AAPL', 'MSFT', 'GOOG')
GROUP BY symbol, month
ORDER BY month, symbol
"
```

### Partitioning

Market data is automatically partitioned by year and symbol in the Parquet files, which enables efficient querying by date ranges and symbols.

### Performance Optimization

For best performance:

1. Store your data in Parquet format using the `vegas ingest` command
2. Query only the data you need using specific date ranges and symbols
3. Use SQL aggregations when possible to reduce data transfer

## Requirements

- DuckDB (>=0.9.0)
- PyArrow (>=14.0.0)
- Pandas (>=1.3.0)

## Examples

See the `examples/database_example.sh` script for a full demonstration of the database system features.

For advanced data analysis using DuckDB, check out `examples/duckdb_analysis_example.py`.

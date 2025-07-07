# Vegas CLI

The Vegas CLI provides a command-line interface for the Vegas backtesting engine.

## Installation

The CLI is automatically installed when you install the Vegas package:

```bash
pip install vegas
```

## Commands

### Run a Backtest

Run a backtest using a strategy file:

```bash
vegas run path/to/strategy.py --start 2018-01-01 --end 2018-12-31 --capital 100000
```

Options:
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--capital`: Initial capital (default: 100000)
- `--data-file`: Path to a single data file
- `--data-dir`: Directory containing data files
- `--output`: Output file for equity curve plot
- `--results-csv`: Output file for results CSV
- `--report`: Generate a QuantStats HTML report and save to the specified file
- `--benchmark`: Benchmark symbol for the QuantStats report (default: SPY)
- `--file-pattern`: Pattern for matching data files (default: *.csv*)
- `--verbose`: Enable verbose logging

### Ingest Data

Ingest data into the database:

```bash
vegas ingest --file path/to/file.csv
```

```bash
vegas ingest --directory path/to/directory --pattern "*.csv"
```

Options:
- `--file`: Path to a single data file
- `--directory`: Directory containing data files
- `--pattern`: Pattern for matching data files (default: *.csv*)
- `--max-files`: Maximum number of files to ingest
- `--data-dir`: Base data directory (default: db)
- `--verbose`: Enable verbose logging

### Ingest OHLCV Data

Ingest OHLCV files into the database:

```bash
vegas ingest-ohlcv --file path/to/file.ohlcv-1h.csv.zst
```

```bash
vegas ingest-ohlcv --directory path/to/directory
```

> **IMPORTANT**: Always use the `ingest-ohlcv` command for OHLCV files, not the general `ingest` command. 
> The `ingest-ohlcv` command is specifically designed to handle the OHLCV file format with the `ts_event` column.

Options:
- `--file`: Path to a single OHLCV file
- `--directory`: Directory containing OHLCV files
- `--max-files`: Maximum number of files to ingest
- `--verbose`: Enable verbose logging

### Database Status

Display database status:

```bash
vegas db-status
```

Options:
- `--detailed`: Show detailed information
- `--show-symbols`: Show list of available symbols
- `--limit`: Limit number of symbols to show
- `--data-dir`: Base data directory (default: db)
- `--verbose`: Enable verbose logging

### Database Query

Run a SQL query on the database:

```bash
vegas db-query --query "SELECT * FROM market_data LIMIT 10"
```

```bash
vegas db-query --file path/to/query.sql
```

Options:
- `--query`: SQL query to execute
- `--file`: File containing SQL query
- `--output`: Output file for query results
- `--data-dir`: Base data directory (default: db)
- `--verbose`: Enable verbose logging

### Delete Database

Delete the database and all parquet files:

```bash
vegas delete-db
```

```bash
vegas delete-db --force
```

Options:
- `--force`, `-f`: Force deletion without confirmation
- `--data-dir`: Base data directory (default: db)
- `--verbose`: Enable verbose logging

## Examples

### Run a Simple Moving Average Strategy

```bash
vegas run examples/simple_ma_strategy.py --start 2018-01-01 --end 2018-12-31
```

### Generate a Performance Report

```bash
vegas run examples/simple_ma_strategy.py --start 2018-01-01 --end 2018-12-31 --report report.html
```

### Use a Different Benchmark

```bash
vegas run examples/simple_ma_strategy.py --start 2018-01-01 --end 2018-12-31 --report report.html --benchmark QQQ
```

### Ingest Data and Run a Backtest

```bash
# Ingest data
vegas ingest --directory data --pattern "*.csv"

# Run backtest using ingested data
vegas run examples/simple_ma_strategy.py --start 2018-01-01 --end 2018-12-31
```

### Ingest OHLCV Data and Run a Backtest

```bash
# Ingest OHLCV data
vegas ingest-ohlcv --directory data

# Run backtest using ingested data
vegas run examples/simple_ma_strategy.py --start 2018-01-01 --end 2018-12-31
```

### Run a Query on the Database

```bash
vegas db-query --query "SELECT symbol, COUNT(*) as count FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 10"
```

### Delete the Database and Start Fresh

```bash
# Delete the database
vegas delete-db

# Ingest data again
vegas ingest-ohlcv --directory data
```

## Database Schema

The Vegas database uses DuckDB and Parquet for efficient storage and querying:

- `market_data`: View of all market data from Parquet files
- `symbols`: Table with available symbols
- `data_sources`: Table tracking ingested data sources

## OHLCV Data Format

The system supports OHLCV (Open, High, Low, Close, Volume) files in the format:
- Filename pattern: `*.ohlcv-1h.csv.zst`
- Compressed with Zstandard
- CSV format with headers: ts_event, symbol, open, high, low, close, volume

Data is automatically partitioned by year, date, and symbol in the Parquet files. 
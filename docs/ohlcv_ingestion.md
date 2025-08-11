# OHLCV Data Ingestion Guide

This guide explains how to ingest OHLCV (Open, High, Low, Close, Volume) data into the Vegas backtesting engine.

## File Format

Vegas supports OHLCV data files in the following format:

- CSV files compressed with Zstandard (`.csv.zst`)
- Expected filename pattern: `[exchange]-[source]-[date].ohlcv-[timeframe].csv.zst`
- Example: `xnas-itch-20250630.ohlcv-1h.csv.zst`

The CSV file should contain the following columns:
- `ts_event`: Timestamp of the candle (ISO format)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `symbol`: Ticker symbol

## Ingestion Methods

### Using the CLI (Recommended)

The Vegas CLI provides a dedicated command for ingesting OHLCV files:

```bash
vegas ingest-ohlcv --file=path/to/file.ohlcv-1h.csv.zst
```

To ingest all OHLCV files in a directory:

```bash
vegas ingest-ohlcv --directory=path/to/directory
```

> **IMPORTANT**: Always use the `ingest-ohlcv` command for OHLCV files, not the general `ingest` command.
> The `ingest-ohlcv` command is specifically designed to handle the OHLCV file format with the `ts_event` column.

### Duplicate Data Prevention

The system automatically prevents duplicate data from being ingested:

- If you try to ingest a file that has already been ingested, it will be skipped
- A notification will be displayed indicating which files were skipped
- This prevents data duplication and ensures data integrity

Example output when trying to ingest already ingested files:

```
OHLCV Ingestion completed: 0 total rows ingested
Skipped 2 files that were already ingested
```

### Limiting the Number of Files

To limit the number of files ingested:

```bash
vegas ingest-ohlcv --directory=path/to/directory --max-files=10
```

### Using Python Directly

You can also ingest OHLCV files directly from Python:

```python
from vegas.data import DataLayer

# Initialize the data layer
data_layer = DataLayer("db")

# Ingest a single file
data_layer.db_manager.ingest_ohlcv_file("path/to/file.ohlcv-1h.csv.zst")

# Ingest all files in a directory
data_layer.db_manager.ingest_ohlcv_directory("path/to/directory")
```

## Storage Format

OHLCV data is stored in Parquet format with the following partitioning scheme:

```
db/
  partitioned/
    year=YYYY/
      month=MM/
        part-0.parquet
        part-1.parquet
        ...
```

This partitioning strategy provides good performance while avoiding hitting partition limits.

## Verifying Ingestion

To verify that your data was ingested correctly, use the `db-status` command:

```bash
vegas db-status --detailed
```

You can also run SQL queries on the ingested data:

```bash
vegas db-query --query "SELECT year, month, COUNT(*) FROM market_data GROUP BY year, month"
```

## Managing the Database

### Deleting the Database

If you need to start fresh or clear all ingested data, you can use the `delete-db` command:

```bash
vegas delete-db
```

This will prompt for confirmation before deleting all database files. To bypass the confirmation prompt:

```bash
vegas delete-db --force
```

This command deletes:
- The DuckDB database file (`db/vegas.duckdb`)
- All Parquet files in the partitioned directory (`db/partitioned`)

After deleting the database, you can ingest data again as if starting from scratch.

## Troubleshooting

### Common Issues

1. **Missing zstandard library**: Install with `pip install zstandard`
2. **Permission denied**: Ensure you have write access to the `db` directory
3. **Empty DataFrame**: Check if your file has the expected column names (`ts_event`, not `timestamp`)
4. **Duplicate data**: If you see "Skipped files that were already ingested", this means those files have already been processed

### Checking File Format

To check if your file has the correct format:

```bash
zstdcat file.ohlcv-1h.csv.zst | head -n 5
```

The first line should show column headers including `ts_event`, `open`, `high`, `low`, `close`, `volume`, and `symbol`.

## Example Script

Here's an example script for ingesting OHLCV files:

```python
# examples/direct_ingest_ohlcv.py
from vegas.data import DataLayer
import os
import glob

# Initialize the data layer
data_layer = DataLayer("db")

# Find all OHLCV files
directory = "data"
files = glob.glob(os.path.join(directory, "*.ohlcv-1h.csv.zst"))

# Ingest each file
for file in files:
    print(f"Ingesting {file}...")
    rows = data_layer.db_manager.ingest_ohlcv_file(file)
    print(f"Ingested {rows} rows from {file}")

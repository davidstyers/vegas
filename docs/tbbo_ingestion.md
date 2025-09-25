# TBBO Tick Data Ingestion Guide

This guide explains how to ingest TBBO (Top of Book Bid/Offer) tick data into the Vegas backtesting engine.

## Overview

TBBO data provides tick-level bid/ask price and size information from market data feeds. This implementation supports the Vegas engine's existing infrastructure with dedicated tick data storage and views.

## File Format

Vegas supports TBBO data files in the following format:

- CSV files compressed with Zstandard (`.tbbo.csv.zst`)
- Expected filename pattern: `[exchange]-[source]-[date].tbbo.csv.zst`
- Example: `xnas-itch-20180501.tbbo.csv.zst`

The CSV file should contain the following columns:
- `ts_event`: Timestamp of the tick (ISO format)
- `symbol`: Ticker symbol
- `bid_px_00`: Top of book bid price
- `ask_px_00`: Top of book ask price
- `bid_sz_00`: Top of book bid size
- `ask_sz_00`: Top of book ask size

## Ingestion Methods

### Using the CLI (Recommended)

The Vegas CLI provides a dedicated command for ingesting TBBO files:

```bash
vegas ingest-tbbo --file=path/to/file.tbbo.csv.zst
```

To ingest all TBBO files in a directory:

```bash
vegas ingest-tbbo --directory=path/to/directory
```

> **IMPORTANT**: Always use the `ingest-tbbo` command for TBBO files, not the general `ingest` command.
> The `ingest-tbbo` command is specifically designed to handle the TBBO file format with proper column mapping.

### Duplicate Data Prevention

The system automatically prevents duplicate data from being ingested:

- If you try to ingest a file that has already been ingested, it will be skipped
- A notification will be displayed indicating which files were skipped
- This prevents data duplication and ensures data integrity

### Limiting the Number of Files

To limit the number of files ingested:

```bash
vegas ingest-tbbo --directory=path/to/directory --max-files=10
```

### Using Python Directly

You can also ingest TBBO files directly from Python:

```python
from vegas.data import DataLayer

# Initialize the data layer
data_layer = DataLayer("db")

# Ingest a single file
data_layer.ingest_tbbo_file("path/to/file.tbbo.csv.zst")

# Ingest all files in a directory
data_layer.ingest_tbbo_directory("path/to/directory")
```

## Storage Format

TBBO data is stored in Parquet format with the following partitioning scheme:

```
db/
  tick_partitioned/
    year=YYYY/
      month=MM/
        part-0.parquet
        part-1.parquet
        ...
```

This partitioning strategy provides good performance while avoiding hitting partition limits.

## Database Schema

TBBO data is stored separately from OHLCV data with its own view:

- `tick_data`: View of all tick data from TBBO Parquet files
- Columns: `timestamp`, `symbol`, `bid_price`, `ask_price`, `bid_size`, `ask_size`

## Verifying Ingestion

To verify that your data was ingested correctly, use the `db-status` command:

```bash
vegas db-status --detailed
```

You can also run SQL queries on the ingested tick data:

```bash
vegas db-query --query "SELECT * FROM tick_data LIMIT 10"
```

## Example Queries

Query tick data for a specific symbol:

```sql
SELECT timestamp, bid_price, ask_price, bid_size, ask_size 
FROM tick_data 
WHERE symbol = 'AAPL' 
ORDER BY timestamp 
LIMIT 100
```

Calculate spread statistics:

```sql
SELECT symbol, 
       AVG(ask_price - bid_price) AS avg_spread,
       MAX(ask_price - bid_price) AS max_spread,
       MIN(ask_price - bid_price) AS min_spread
FROM tick_data 
WHERE timestamp >= '2018-05-01' AND timestamp < '2018-05-02'
GROUP BY symbol
ORDER BY avg_spread DESC
```

## Integration with Engine

The tick data can be accessed through the standard data portal and engine interfaces. The tick data is automatically available once ingested and can be queried alongside OHLCV data.

## Column Mapping

The system automatically maps TBBO file columns to the internal schema:

| TBBO File Column | Internal Column |
|------------------|-----------------|
| `ts_event`       | `timestamp`     |
| `bid_px_00`      | `bid_price`     |
| `ask_px_00`      | `ask_price`     |
| `bid_sz_00`      | `bid_size`      |
| `ask_sz_00`      | `ask_size`      |
| `symbol`         | `symbol`        |

## Performance Considerations

- TBBO tick data can be very large. The partitioned storage helps with query performance.
- Use date range filters in queries to limit the amount of data processed.
- The system uses separate storage for tick data to avoid impacting OHLCV data performance.

## Troubleshooting

### Common Issues

1. **Missing zstandard library**: Install with `pip install zstandard`
2. **Permission denied**: Ensure you have write access to the `db` directory
3. **Empty DataFrame**: Check if your file has the expected column names (`bid_px_00`, `ask_px_00`, etc.)
4. **Duplicate data**: If you see "Skipped files that were already ingested", this means those files have already been processed

### Checking File Format

To check if your file has the correct format:

```bash
zstdcat file.tbbo.csv.zst | head -n 5
```

The first line should show column headers including `ts_event`, `symbol`, `bid_px_00`, `ask_px_00`, `bid_sz_00`, and `ask_sz_00`.

## Managing the Database

### Deleting the Database

If you need to start fresh or clear all ingested data, you can use the `delete-db` command:

```bash
vegas delete-db
```

This will delete both OHLCV and TBBO data.

## Example Usage

```bash
# Ingest TBBO files from a directory
vegas ingest-tbbo --directory=data --max-files=5

# Check database status
vegas db-status --detailed

# Query the tick data
vegas db-query --query "SELECT symbol, COUNT(*) as tick_count FROM tick_data GROUP BY symbol ORDER BY tick_count DESC LIMIT 10"

# Export tick data to CSV
vegas db-query --query "SELECT * FROM tick_data WHERE symbol = 'SPY' LIMIT 1000" --output spy_ticks.csv
```

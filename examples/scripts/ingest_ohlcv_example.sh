#!/bin/bash
# Example script demonstrating how to ingest OHLCV files into the database

# Set up paths
DATA_DIR="data"
DB_DIR="db"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

# Create db directory if it doesn't exist
mkdir -p $DB_DIR/partitioned

echo "================================================"
echo "Vegas OHLCV Data Ingestion Example"
echo "================================================"

# Step 1: Ingest OHLCV files into the database
echo
echo "Step 1: Ingesting OHLCV files into the database..."
echo "Command: vegas ingest-ohlcv --directory $DATA_DIR --max-files 5"
echo
vegas ingest-ohlcv --directory $DATA_DIR --max-files 5

# Step 2: Check database status
echo
echo "Step 2: Checking database status..."
echo "Command: vegas db-status --detailed"
echo
vegas db-status --detailed

# Step 3: Run a SQL query on the database
echo
echo "Step 3: Running a SQL query on the database..."
echo "Command: vegas db-query --query \"SELECT symbol, COUNT(*) as count, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 10\""
echo
vegas db-query --query "SELECT symbol, COUNT(*) as count, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 10"

# Step 4: Run a backtest using the ingested data
echo
echo "Step 4: Running a backtest using the ingested data..."
echo "Command: vegas run examples/simple_ma_strategy.py --start 2018-05-01 --end 2018-05-31"
echo
vegas run examples/simple_ma_strategy.py --start 2018-05-01 --end 2018-05-31

echo
echo "OHLCV ingestion example completed!"
echo "The Vegas backtesting engine now stores OHLCV data in Parquet format in the db directory."
echo "Data is partitioned by year, month, and symbol for efficient querying."

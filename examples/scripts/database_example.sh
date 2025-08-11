#!/bin/bash
# Example script demonstrating the DuckDB and Parquet database features

# Set up paths
DATA_DIR="data"
SAMPLE_DATA="data/sample_data.csv.zst"

# Check if sample data exists
if [ ! -f "$SAMPLE_DATA" ]; then
    echo "Sample data file not found: $SAMPLE_DATA"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

echo "================================================"
echo "Vegas DuckDB & Parquet Database Example"
echo "================================================"

# Step 1: Ingest data into the database
echo
echo "Step 1: Ingesting data into the database..."
echo "Command: vegas ingest --file $SAMPLE_DATA"
echo
vegas ingest --file $SAMPLE_DATA

# Step 2: Check database status
echo
echo "Step 2: Checking database status..."
echo "Command: vegas db-status --detailed"
echo
vegas db-status --detailed

# Step 3: Run a SQL query on the database
echo
echo "Step 3: Running a SQL query on the database..."
echo "Command: vegas db-query --query \"SELECT symbol, COUNT(*) as count FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 5\""
echo
vegas db-query --query "SELECT symbol, COUNT(*) as count FROM market_data GROUP BY symbol ORDER BY count DESC LIMIT 5"

# Step 4: Run a backtest using database data
echo
echo "Step 4: Running a backtest using database data..."
echo "Command: vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01"
echo
vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01

echo
echo "Database example completed!"
echo "The Vegas backtesting engine can now use DuckDB and Parquet for efficient data storage and querying."
echo "Data is automatically stored in Parquet format and queried using DuckDB."

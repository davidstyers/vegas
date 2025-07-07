#!/bin/bash
# Example script to demonstrate using the Vegas CLI with already ingested data

# First, ingest some sample data (this could be done separately)
echo "Step 1: Ingesting sample data..."
vegas run examples/simple_ma_strategy.py --data-file data/sample_data.csv.zst --start 2020-01-01 --end 2020-01-02

# Now run without specifying a data file, using the already ingested data
echo
echo "Step 2: Running backtest using already ingested data..."
echo "Notice that we don't specify --data-file or --data-dir"
echo
vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01 --output equity_curve_ingested_data.png

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo
    echo "Backtest with ingested data completed successfully!"
    echo "Results saved to: equity_curve_ingested_data.png"
    echo
    echo "This demonstrates how Vegas can use data that was previously ingested,"
    echo "without needing to specify the data source each time."
else
    echo
    echo "Backtest failed. Check the error messages above."
fi 
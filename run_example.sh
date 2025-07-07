#!/bin/bash
# Run the example strategy

# Check if the strategy file exists
if [ ! -f "examples/simple_ma_strategy.py" ]; then
    echo "Strategy file not found: examples/simple_ma_strategy.py"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

# Run the strategy with sample data
echo "Running example strategy with sample data..."
vegas run examples/simple_ma_strategy.py --data-file data/sample_data.csv.zst --start 2020-01-01 --end 2021-01-01 --output equity_curve.png --results-csv results.csv

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo
    echo "Backtest completed successfully!"
    echo "Results saved to:"
    echo "- Equity curve: equity_curve.png"
    echo "- CSV data: results.csv"
    echo
    echo "To run with different parameters, try:"
    echo "vegas run examples/simple_ma_strategy.py --data-file data/sample_data.csv.zst --start 2021-01-01 --end 2022-01-01"
    echo
    echo "Or run using already ingested data (without specifying a data file):"
    echo "vegas run examples/simple_ma_strategy.py --start 2020-01-01 --end 2021-01-01"
else
    echo
    echo "Backtest failed. Check the error messages above."
fi 
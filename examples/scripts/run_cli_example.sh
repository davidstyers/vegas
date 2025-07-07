#!/bin/bash
# Example script demonstrating how to use the Vegas CLI

# Set up variables
STRATEGY_FILE="simple_ma_strategy.py"
DATA_FILE="../data/sample_data.csv.zst"
DATA_DIR="../data/us-equities"
START_DATE="2020-01-01"
END_DATE="2021-01-01"
OUTPUT_FILE="equity_curve.png"
RESULTS_CSV="results.csv"

# Print header
echo "===== Vegas CLI Examples ====="
echo

# Check if the strategy file exists
if [ ! -f "$STRATEGY_FILE" ]; then
    echo "Strategy file not found: $STRATEGY_FILE"
    echo "Make sure you're running this script from the examples directory."
    exit 1
fi

# Example 1: Basic usage with a data file
echo "Example 1: Running backtest with a data file"
echo "Command: vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE"
echo
vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE
echo

# Example 2: Save equity curve plot
echo "Example 2: Saving equity curve plot"
echo "Command: vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE --output $OUTPUT_FILE"
echo
vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE --output $OUTPUT_FILE
echo

# Example 3: Save results to CSV
echo "Example 3: Saving results to CSV"
echo "Command: vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE --results-csv $RESULTS_CSV"
echo
vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE --results-csv $RESULTS_CSV
echo

# Example 4: Run with data from a directory (if available)
if [ -d "$DATA_DIR" ]; then
    echo "Example 4: Running with data from a directory"
    echo "Command: vegas run $STRATEGY_FILE --data-dir $DATA_DIR --start $START_DATE --end $END_DATE"
    echo
    vegas run $STRATEGY_FILE --data-dir $DATA_DIR --start $START_DATE --end $END_DATE
    echo
fi

# Example 5: Enable verbose logging
echo "Example 5: Enabling verbose logging"
echo "Command: vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE -v"
echo
vegas run $STRATEGY_FILE --data-file $DATA_FILE --start $START_DATE --end $END_DATE -v
echo

echo "===== End of Examples ====="

# Display results
echo
echo "Results:"
echo "- Equity curve plot: $OUTPUT_FILE"
echo "- Results CSV: $RESULTS_CSV"
echo

# Make the script executable
chmod +x run_cli_example.sh 
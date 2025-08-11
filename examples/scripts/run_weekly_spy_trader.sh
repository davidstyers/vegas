#!/bin/bash
# Script to run the Weekly SPY Trading Strategy example

# Change to the project root directory
cd "$(dirname "$0")/../.." || exit

# Default settings
DATA_DIR="db"
START_DATE="2022-01-01"
END_DATE="2022-12-31"
INITIAL_CAPITAL="100000.0"

# Help message
function show_help {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --data-dir DIR      Directory with market data (default: $DATA_DIR)"
  echo "  --start DATE        Start date in YYYY-MM-DD format (default: $START_DATE)"
  echo "  --end DATE          End date in YYYY-MM-DD format (default: $END_DATE)"
  echo "  --capital AMOUNT    Initial capital (default: $INITIAL_CAPITAL)"
  echo "  -h, --help          Show this help message"
  echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --start)
      START_DATE="$2"
      shift 2
      ;;
    --end)
      END_DATE="$2"
      shift 2
      ;;
    --capital)
      INITIAL_CAPITAL="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Make sure python environment is activated
source "$(pwd)/venv/bin/activate" 2>/dev/null || echo "Warning: Could not activate virtual environment. Make sure to run in the correct environment."

# Run the Weekly SPY Trading Strategy
python examples/weekly_spy_trader.py \
  --data-dir "$DATA_DIR" \
  --start "$START_DATE" \
  --end "$END_DATE" \
  --initial-capital "$INITIAL_CAPITAL"

# Exit with the same exit code as the Python script
exit $?

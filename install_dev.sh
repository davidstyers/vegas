#!/bin/bash
# Install Vegas in development mode

# Set up a virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
else
    echo "Using existing virtual environment..."
    source venv/bin/activate
fi

# Install the package in development mode
echo "Installing Vegas in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest

echo "Installation complete!"
echo "You can now use the 'vegas' command to run backtests."
echo
echo "Example usage:"
echo "vegas run examples/simple_ma_strategy.py --data-file data/sample_data.csv.zst --start 2020-01-01 --end 2021-01-01"
echo
echo "For help, run:"
echo "vegas --help"

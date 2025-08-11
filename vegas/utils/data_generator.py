"""Data generation utilities for the Vegas backtesting engine.

This module provides functionality for generating synthetic market data
for testing and development purposes.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import zstandard as zstd


def generate_synthetic_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    freq: str = "1H",
    base_price: float = 100.0,
    volatility: float = 0.01,
    seed: Optional[int] = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for a list of symbols.

    :param symbols: Symbols to generate data for.
    :type symbols: list[str]
    :param start_date: Start datetime for the series.
    :type start_date: datetime
    :param end_date: End datetime for the series.
    :type end_date: datetime
    :param freq: Pandas frequency string (e.g., '1H' for hourly).
    :type freq: str
    :param base_price: Base price around which random walk evolves.
    :type base_price: float
    :param volatility: Per-period volatility used for returns.
    :type volatility: float
    :param seed: Random seed for reproducibility.
    :type seed: Optional[int]
    :returns: Mapping symbol -> pandas DataFrame of OHLCV.
    :rtype: Dict[str, pandas.DataFrame]
    :Example:
        >>> data = generate_synthetic_data(['AAPL'], datetime(2022,1,1), datetime(2022,1,31))
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_dates = len(dates)

    # Dictionary to hold DataFrames
    data_dict = {}

    for i, symbol in enumerate(symbols):
        # Use different base prices for different symbols
        symbol_base = base_price * (0.8 + 0.4 * (i / len(symbols)))

        # Generate random walk for close prices
        daily_returns = np.random.normal(0, volatility, n_dates)
        close_prices = symbol_base * np.cumprod(1 + daily_returns)

        # Generate OHLCV data
        opens = close_prices * np.random.uniform(0.99, 1.01, n_dates)
        highs = np.maximum(close_prices * np.random.uniform(1.0, 1.02, n_dates), opens)
        lows = np.minimum(close_prices * np.random.uniform(0.98, 1.0, n_dates), opens)
        volumes = np.random.randint(1000, 100000, n_dates)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": close_prices,
                "volume": volumes,
            }
        )

        data_dict[symbol] = df

    return data_dict


def save_to_csv_zstd(data_dict: Dict[str, pd.DataFrame], output_path: str) -> None:
    """Save synthetic data to a Zstandard-compressed CSV file.

    :param data_dict: Mapping symbol -> pandas DataFrame with OHLCV data.
    :type data_dict: Dict[str, pd.DataFrame]
    :param output_path: Destination path for compressed CSV.
    :type output_path: str
    :returns: None
    :rtype: None
    """
    # Combine all DataFrames
    combined_df = pd.concat(data_dict.values(), ignore_index=True)

    # Sort by timestamp
    combined_df = combined_df.sort_values("timestamp")

    # Create parent directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert to CSV
    csv_data = combined_df.to_csv(index=False)

    # Compress with Zstandard
    with open(output_path, "wb") as f:
        compressor = zstd.ZstdCompressor(level=10)
        compressed_data = compressor.compress(csv_data.encode("utf-8"))
        f.write(compressed_data)


def generate_sample_dataset(output_path: str = "data/sample_data.csv.zst") -> None:
    """Generate a sample dataset for testing and write to `output_path`.

    :param output_path: Destination path for compressed CSV.
    :type output_path: str
    :returns: None
    :rtype: None
    """
    # Define parameters
    symbols = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOG",
        "META",
        "TSLA",
        "NVDA",
        "PYPL",
        "NFLX",
        "ADBE",
        "INTC",
        "CSCO",
    ]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)

    # Generate data
    print(
        f"Generating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}..."
    )
    data_dict = generate_synthetic_data(symbols, start_date, end_date)

    # Save to file
    print(f"Saving data to {output_path}...")
    save_to_csv_zstd(data_dict, output_path)

    # Print summary
    total_rows = sum(len(df) for df in data_dict.values())
    print(f"Generated {total_rows} rows of data for {len(symbols)} symbols")
    print(f"Data saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # Generate sample data when run directly
    generate_sample_dataset()

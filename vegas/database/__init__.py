"""Database Layer for the Vegas backtesting engine.

This module provides functionality for managing market data using DuckDB and Parquet files.
"""

from vegas.database.database import DatabaseManager, ParquetManager

__all__ = ["DatabaseManager", "ParquetManager"] 
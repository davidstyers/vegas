"""Database implementation for the Vegas backtesting engine."""

from vegas.database.database import DatabaseManager, ParquetManager

# Re-export public API, including trading days helper
__all__ = ["DatabaseManager", "ParquetManager"]
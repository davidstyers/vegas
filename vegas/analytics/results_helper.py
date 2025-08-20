"""Helper functions for creating Results objects from backtest data."""

import polars as pl
from typing import Any, Dict

from vegas.analytics.analytics import Results


def create_results_from_dict(results_dict: Dict[str, Any]) -> Results:
    """Create a Results instance from a backtest results dictionary.
    
    This function converts the dictionary format returned by the engine
    into a Results object for easier analysis and reporting.
    
    Args:
        results_dict: Dictionary containing backtest results
        
    Returns:
        Results object with all backtest data
        
    Example:
        >>> results = engine.run(start, end, strategy, initial_capital)
        >>> results_obj = create_results_from_dict(results)
        >>> results_obj.create_tearsheet(output_file="report.html")
    """
    # Extract data from the results dictionary
    equity_curve = results_dict.get("equity_curve", pl.DataFrame())
    trades = results_dict.get("transactions", pl.DataFrame())  # Note: engine uses "transactions"
    stats = results_dict.get("stats", {})
    positions_history = results_dict.get("positions_history", {})
    returns = results_dict.get("returns", pl.DataFrame())
    
    # If returns is not provided, calculate it from equity curve
    if returns.is_empty() and not equity_curve.is_empty():
        returns = _calculate_returns_from_equity_curve(equity_curve)
    
    return Results(
        equity_curve=equity_curve,
        trades=trades,
        stats=stats,
        positions_history=positions_history,
        returns=returns,
    )


def _calculate_returns_from_equity_curve(equity_curve: pl.DataFrame) -> pl.DataFrame:
    """Calculate returns from equity curve data.
    
    Args:
        equity_curve: DataFrame with timestamp and equity columns
        
    Returns:
        DataFrame with timestamp and return columns
    """
    if equity_curve.is_empty() or "equity" not in equity_curve.columns:
        return pl.DataFrame(schema={"timestamp": pl.Datetime, "return": pl.Float64})
    
    return equity_curve.with_columns(
        pl.col("equity").pct_change().fill_null(0).alias("return")
    ).select(["timestamp", "return"])

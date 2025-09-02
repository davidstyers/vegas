"""Decision logic for the Alpha submodule."""

from typing import Dict, Any


def make_decision(metrics: Dict[str, Any]) -> bool:
    """Applies a set of rules to the calculated metrics to decide if a strategy is promising.

    Args:
        metrics: A dictionary of performance metrics from a backtest run.

    Returns:
        True if the strategy is promising, False otherwise.
    """
    # Example decision logic:
    # - Total return must be positive
    # - Sharpe ratio must be greater than 1.0
    # - Max drawdown must be less than 20%
    total_return = metrics.get("total_return_pct", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    max_drawdown = metrics.get("max_drawdown", 0.0)

    if total_return > 0 and sharpe > 1.0 and max_drawdown < 0.2:
        return True
    return False
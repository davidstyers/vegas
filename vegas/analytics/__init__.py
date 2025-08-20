"""Analytics and Statistics Layer for the Vegas backtesting engine."""

from vegas.analytics.analytics import Analytics, Results
from vegas.analytics.results_helper import create_results_from_dict
from vegas.analytics.hyperopt import (
    evaluate_strategy_params,
    create_optuna_objective,
    optimize_strategy,
    run_optimized_backtest,
    print_optimization_summary,
    OptimizationConfig,
)

__all__ = [
    "Analytics", 
    "Results",
    "create_results_from_dict",
    "evaluate_strategy_params", 
    "create_optuna_objective",
    "optimize_strategy",
    "run_optimized_backtest",
    "print_optimization_summary",
    "OptimizationConfig",
]

"""Example demonstrating hyperparameter optimization with the Vegas analytics library.

This example shows how to use the refactored hyperparameter optimization functions
from vegas.analytics to optimize strategy parameters.
"""

import logging
from datetime import datetime

from vegas.analytics import (
    configure_logging,
    optimize_strategy,
    run_optimized_backtest,
    print_optimization_summary,
)
from vegas.strategy import Strategy


class SimpleStrategy(Strategy):
    """Simple example strategy for demonstration purposes."""
    
    def __init__(self, param1: float, param2: float):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def initialize(self, context):
        context.param1 = self.param1
        context.param2 = self.param2
        context.logger = logging.getLogger(__name__)
    
    def handle_data(self, context, data):
        # Simple example logic
        context.logger.info(f"Strategy running with params: {self.param1}, {self.param2}")
        return []


def main():
    """Demonstrate hyperparameter optimization functionality."""
    
    # Configure logging
    configure_logging(verbose=True)
    logger = logging.getLogger(__name__)
    
    # Define backtest parameters
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)
    initial_capital = 100000.0
    
    # Strategy factory function
    def create_strategy(**params):
        return SimpleStrategy(
            param1=params.get("param1", 0.5),
            param2=params.get("param2", 0.5),
        )
    
    # Custom parameter suggestions
    param_suggestions = {
        "param1": lambda trial: trial.suggest_float("param1", 0.0, 1.0),
        "param2": lambda trial: trial.suggest_float("param2", 0.0, 1.0),
    }
    
    logger.info("Starting hyperparameter optimization example...")
    
    # Example 1: Run optimization and get best parameters
    logger.info("\n=== Example 1: Basic Optimization ===")
    best_params, study = optimize_strategy(
        strategy_factory=create_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        n_trials=5,  # Small number for demo
        study_name="example_optimization",
        logger=logger,
        param_suggestions=param_suggestions,
    )
    
    logger.info(f"Best parameters found: {best_params}")
    print_optimization_summary(study, logger)
    
    # Example 2: Run optimized backtest
    logger.info("\n=== Example 2: Optimized Backtest ===")
    results = run_optimized_backtest(
        strategy_factory=create_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        optimize=True,
        n_trials=3,  # Small number for demo
        study_name="example_backtest",
        logger=logger,
        param_suggestions=param_suggestions,
    )
    
    logger.info(f"Backtest results: {results.get('stats', {})}")
    
    # Example 3: Run single backtest without optimization
    logger.info("\n=== Example 3: Single Backtest ===")
    default_params = {"param1": 0.3, "param2": 0.7}
    results = run_optimized_backtest(
        strategy_factory=create_strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        optimize=False,
        default_params=default_params,
        logger=logger,
    )
    
    logger.info(f"Single backtest results: {results.get('stats', {})}")
    
    logger.info("Hyperparameter optimization example completed!")


if __name__ == "__main__":
    main()

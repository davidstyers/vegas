#!/usr/bin/env python3
"""Examples showing different ways to configure adaptive hyperparameter optimization.

This file demonstrates various configuration approaches for the refactored hyperopt module,
including fixed vs. optimized engine parameters and different parameter types.
"""

import logging
from datetime import datetime
from typing import List

from vegas.analytics.hyperopt import (
    OptimizationConfig,
    optimize_strategy,
    create_engine_suggestions,
    create_frequency_suggestions,
    create_data_type_suggestions,
    create_calendar_suggestions,
)
from vegas.strategy import Strategy, Context, Signal


class ExampleStrategy(Strategy):
    """Example strategy for demonstration."""
    
    def initialize(self, context: Context) -> None:
        """Initialize the strategy."""
        context.symbols = ["AAPL", "MSFT"]
        context.window = getattr(context, "window", 20)
        
    def handle_data(self, context: Context, data) -> List[Signal]:
        """Generate simple trading signals."""
        signals = []
        
        for symbol in context.symbols:
            if not data.has_data(symbol):
                continue
                
            prices = data.history(symbol, "close", context.window + 1)
            if len(prices) < context.window:
                continue
                
            # Simple momentum signal
            if prices.iloc[-1] > prices.iloc[-context.window]:
                signals.append(Signal(symbol=symbol, quantity=100))
            else:
                signals.append(Signal(symbol=symbol, quantity=-100))
        
        return signals


def strategy_factory(window: int = 20) -> Strategy:
    """Factory function to create strategy instances."""
    strategy = ExampleStrategy()
    strategy.window = window
    return strategy


def example_1_fixed_engine_params():
    """Example 1: Fixed engine parameters, only optimize strategy parameters."""
    print("\n=== Example 1: Fixed Engine Parameters ===")
    
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "window": {"type": "int", "low": 10, "high": 50},
        },
        # Fixed engine parameters
        frequency="1h",
        data_type="ohlcv", 
        calendar="NYSE",
        objective_metric="sharpe_ratio",
        n_trials=10,
        study_name="fixed_engine_example",
        direction="maximize",
    )
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    print(f"Best strategy parameters: {best_params}")
    print(f"Fixed engine parameters: frequency=1h, data_type=ohlcv, calendar=NYSE")


def example_2_optimize_frequency_only():
    """Example 2: Optimize frequency only, fix other engine parameters."""
    print("\n=== Example 2: Optimize Frequency Only ===")
    
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "window": {"type": "int", "low": 10, "high": 50},
        },
        # Fixed engine parameters
        data_type="ohlcv",
        calendar="NYSE",
        objective_metric="sharpe_ratio",
        n_trials=10,
        study_name="frequency_optimization_example",
        direction="maximize",
    )
    
    # Only optimize frequency
    engine_suggestions = create_frequency_suggestions(["1h", "4h", "1d"])
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        engine_param_suggestions=engine_suggestions,
    )
    
    print(f"Best parameters: {best_params}")


def example_3_optimize_calendar_only():
    """Example 3: Optimize calendar only, fix other engine parameters."""
    print("\n=== Example 3: Optimize Calendar Only ===")
    
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "window": {"type": "int", "low": 10, "high": 50},
        },
        # Fixed engine parameters
        frequency="1h",
        data_type="ohlcv",
        objective_metric="sharpe_ratio",
        n_trials=10,
        study_name="calendar_optimization_example",
        direction="maximize",
    )
    
    # Only optimize calendar
    engine_suggestions = create_calendar_suggestions(["NYSE", "24/7", "24/7_EST"])
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        engine_param_suggestions=engine_suggestions,
    )
    
    print(f"Best parameters: {best_params}")


def example_4_optimize_all_engine_params():
    """Example 4: Optimize all engine parameters."""
    print("\n=== Example 4: Optimize All Engine Parameters ===")
    
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "window": {"type": "int", "low": 10, "high": 50},
        },
        objective_metric="sharpe_ratio",
        n_trials=15,
        study_name="all_engine_optimization_example",
        direction="maximize",
    )
    
    # Optimize all engine parameters
    engine_suggestions = create_engine_suggestions(
        frequencies=["1h", "4h", "1d"],
        data_types=["ohlcv"],
        calendars=["NYSE", "24/7"],
    )
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        engine_param_suggestions=engine_suggestions,
    )
    
    print(f"Best parameters: {best_params}")


def example_5_using_config_ranges():
    """Example 5: Using config ranges instead of suggestions."""
    print("\n=== Example 5: Using Config Ranges ===")
    
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "window": {"type": "int", "low": 10, "high": 50},
        },
        # Define engine parameter ranges in config
        frequency_ranges={
            "frequency": {"type": "categorical", "choices": ["1h", "4h", "1d"]}
        },
        calendar_ranges={
            "calendar": {"type": "categorical", "choices": ["NYSE", "24/7"]}
        },
        # Fixed data_type
        data_type="ohlcv",
        objective_metric="sharpe_ratio",
        n_trials=10,
        study_name="config_ranges_example",
        direction="maximize",
    )
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    print(f"Best parameters: {best_params}")


def main():
    """Run all examples."""
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    
    print("Adaptive Hyperparameter Optimization Examples")
    print("=" * 50)
    
    # Run examples
    example_1_fixed_engine_params()
    example_2_optimize_frequency_only()
    example_3_optimize_calendar_only()
    example_4_optimize_all_engine_params()
    example_5_using_config_ranges()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()

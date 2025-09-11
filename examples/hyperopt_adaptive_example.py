#!/usr/bin/env python3
"""Example demonstrating adaptive hyperparameter optimization with frequency, data_type, and calendar support.

This example shows how to use the refactored hyperopt module to optimize not just strategy parameters,
but also engine configuration parameters like frequency, data_type, and calendar.
"""

import logging
from datetime import datetime
from typing import List

from vegas.analytics.hyperopt import (
    OptimizationConfig,
    optimize_strategy,
    create_engine_suggestions,
    print_optimization_summary,
)
from vegas.strategy import Strategy, Context, Signal


class AdaptiveMovingAverageStrategy(Strategy):
    """Simple moving average strategy for demonstration."""
    
    def initialize(self, context: Context) -> None:
        """Initialize the strategy."""
        context.symbols = ["AAPL", "MSFT", "GOOGL"]
        context.short_window = getattr(context, "short_window", 10)
        context.long_window = getattr(context, "long_window", 30)
        context.position_size = getattr(context, "position_size", 100)
        
    def handle_data(self, context: Context, data) -> List[Signal]:
        """Generate trading signals based on moving averages."""
        signals = []
        
        for symbol in context.symbols:
            if not data.has_data(symbol):
                continue
                
            # Get price data
            prices = data.history(symbol, "close", context.long_window + 1)
            if len(prices) < context.long_window:
                continue
                
            # Calculate moving averages
            short_ma = prices.tail(context.short_window).mean()
            long_ma = prices.tail(context.long_window).mean()
            current_price = prices.iloc[-1]
            
            # Generate signals
            if short_ma > long_ma and current_price > short_ma:
                # Buy signal
                signals.append(Signal(symbol=symbol, quantity=context.position_size))
            elif short_ma < long_ma and current_price < short_ma:
                # Sell signal
                signals.append(Signal(symbol=symbol, quantity=-context.position_size))
        
        return signals


def strategy_factory(short_window: int = 10, long_window: int = 30, position_size: int = 100) -> Strategy:
    """Factory function to create strategy instances."""
    strategy = AdaptiveMovingAverageStrategy()
    strategy.short_window = short_window
    strategy.long_window = long_window
    strategy.position_size = position_size
    return strategy


def main():
    """Main function demonstrating adaptive hyperparameter optimization."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Define optimization configuration
    config = OptimizationConfig(
        initial_capital=100_000.0,
        param_ranges={
            "short_window": {"type": "int", "low": 5, "high": 20},
            "long_window": {"type": "int", "low": 20, "high": 50},
            "position_size": {"type": "int", "low": 50, "high": 200},
        },
        objective_metric="sharpe_ratio",
        n_trials=20,  # Reduced for demo
        study_name="adaptive_ma_optimization",
        direction="maximize",
        n_jobs=1,
    )
    
    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    
    # Create engine parameter suggestions
    engine_suggestions = create_engine_suggestions(
        frequencies=["1h", "4h", "1d"],  # Test different frequencies
        data_types=["ohlcv"],  # Only OHLCV for this example
        calendars=["NYSE", "24/7"],  # Test different calendars
    )
    
    logger.info("Starting adaptive hyperparameter optimization...")
    logger.info(f"Strategy parameters: {list(config.param_ranges.keys())}")
    logger.info(f"Engine parameters: {list(engine_suggestions.keys())}")
    
    # Run optimization
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        logger=logger,
        engine_param_suggestions=engine_suggestions,
    )
    
    # Print results
    print_optimization_summary(study, logger)
    
    logger.info("Optimization completed!")
    logger.info(f"Best parameters: {best_params}")
    
    # Separate strategy and engine parameters for clarity
    strategy_params = {k: v for k, v in best_params.items() 
                      if k in config.param_ranges}
    engine_params = {k: v for k, v in best_params.items() 
                    if k in engine_suggestions}
    
    logger.info(f"Best strategy parameters: {strategy_params}")
    logger.info(f"Best engine parameters: {engine_params}")


if __name__ == "__main__":
    main()

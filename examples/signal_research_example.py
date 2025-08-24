#!/usr/bin/env python3
"""
Signal Research Extension Example

This example demonstrates how to use the signal research mode
to evaluate predictive power of strategy-generated signals.
"""

from datetime import datetime
from typing import Dict

import polars as pl
import numpy as np

from vegas.engine.engine import BacktestEngine
from vegas.strategy import Strategy
from vegas.analytics.alpha import Alpha


class MeanReversionStrategy(Strategy):
    """
    Example strategy that generates signals based on mean reversion.
    
    The strategy looks for assets that have deviated significantly
    from their moving average and bets on reversion to the mean.
    """
    
    def __init__(self, lookback_window: int = 20, z_threshold: float = 2.0):
        super().__init__()
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold
        self.universe = ["AAPL", "MSFT", "TSLA"]
    
    def predict(self, t: int, data: Dict[str, pl.DataFrame]) -> Dict[str, float]:
        """Generate mean reversion signals."""
        signals = {}
        
        for symbol, df in data.items():
            if df.height >= self.lookback_window:
                # Get recent close prices
                close_prices = df.select("close").to_series().to_list()
                recent_prices = close_prices[-self.lookback_window:]
                
                # Calculate rolling statistics
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                
                if std_price > 0:
                    # Calculate z-score of current price
                    current_price = recent_prices[-1]
                    z_score = (current_price - mean_price) / std_price
                    
                    # Generate mean reversion signal
                    # Negative when price is high (expect reversion down)
                    # Positive when price is low (expect reversion up)
                    if abs(z_score) > self.z_threshold:
                        signals[symbol] = -z_score  # Contrarian signal
                    else:
                        signals[symbol] = 0.0  # No signal
        
        return signals


def main():
    """Demonstrate the signal research workflow."""
    print("=== Vegas Signal Research Extension Demo ===\n")
    
    # Setup the backtesting engine with mock data
    engine = BacktestEngine()
    
    # Create our strategy
    strategy = MeanReversionStrategy(lookback_window=20, z_threshold=1.5)
    print(f"   Strategy: {strategy.__class__.__name__}")
    print(f"   Universe: {strategy.universe}")
    print(f"   Parameters: lookback={strategy.lookback_window}, threshold={strategy.z_threshold}\n")
    
    # Define analysis period
    start = datetime(2024, 3, 1, 10)
    end = datetime(2024, 3, 15, 16)  # 2 weeks of data
    print(f"2. Analysis period: {start} to {end}\n")
    
    # Generate signals using the new signal research mode
    print("3. Generating signals...")
    signals_df = engine.generate_signals(start, end, strategy)
    
    print(f"   Generated signals for {signals_df.height} timestamps")
    print(f"   Columns: {signals_df.columns}")
    print(f"   Sample signals:")
    print(signals_df.head(3))
    print()
    
    # Get price data for alpha evaluation
    print("4. Preparing price data for alpha evaluation...")
    engine.data_portal.load_data(start, end, symbols=strategy.universe, frequencies=["1h"])
    
    # Get all historical price data at once
    all_price_data = engine.data_portal.history(
        assets=strategy.universe, 
        bar_count=10000,  # Large number to get all data
        frequency="1h", 
        end_dt=end
    )
    
    if all_price_data.is_empty():
        print("   Warning: No price data available!")
        return
    
    # Pivot the data to get datetime x assets format
    prices_df = all_price_data.pivot(
        index="timestamp",
        on="symbol", 
        values="close"
    ).rename({"timestamp": "datetime"})
    
    print(f"   Price data shape: {prices_df.shape}")
    print(f"   Sample prices:")
    print(prices_df.head(3))
    print()
    
    # Evaluate alpha using the Alpha class
    print("5. Evaluating predictive power (Alpha analysis)...")
    alpha = Alpha(signals_df, prices_df)
    
    # Calculate forward returns for multiple horizons
    horizons = [1, 5, 20]  # 1 hour, 5 hours, 20 hours ahead
    fwd_returns = alpha.forward_returns(horizons=horizons)
    
    print(f"   Computed forward returns for horizons: {horizons}")
    for h in horizons:
        print(f"   {h}-period forward returns shape: {fwd_returns[h].shape}")
    
    # Evaluate signal quality
    evaluation = alpha.evaluate(horizons=horizons)
    print(f"\n   Alpha evaluation results:")
    print(evaluation)
    print()
    
    # Interpret results
    print("6. Interpretation:")
    for row in evaluation.iter_rows(named=True):
        horizon = row["horizon"]
        ic = row["IC"]
        hit_rate = row["HitRate"]
        
        print(f"   {horizon}-period horizon:")
        print(f"     • Information Coefficient (IC): {ic:.4f}")
        print(f"       {'Strong' if abs(ic) > 0.05 else 'Weak'} {'positive' if ic > 0 else 'negative'} correlation")
        print(f"     • Hit Rate: {hit_rate:.1%}")
        print(f"       {'Above' if hit_rate > 0.5 else 'Below'} random (50%)")
        print()
    
    # Summary
    mean_ic = evaluation["IC"].mean()
    mean_hit_rate = evaluation["HitRate"].mean()
    
    print(f"7. Overall Assessment:")
    print(f"   Average IC across horizons: {mean_ic:.4f}")
    print(f"   Average Hit Rate: {mean_hit_rate:.1%}")
    
    if abs(mean_ic) > 0.02 and mean_hit_rate > 0.52:
        print("   Strategy shows promise - consider full backtesting")
    elif abs(mean_ic) > 0.01 or mean_hit_rate > 0.51:
        print("   Strategy shows weak signal - refinement needed")
    else:
        print("   Strategy shows little predictive power - major changes required")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tick Bar Backtesting Example

This example demonstrates how to use Vegas' tick bar capabilities for algorithmic
trading strategy development. The strategy analyzes price momentum using tick bars
created from tick data aggregation and makes trading decisions based on price movements.

Strategy Logic:
- Create tick bars from aggregated tick data (1000 ticks per bar)
- Analyze price momentum over recent tick bars
- Enter long when strong upward momentum is detected
- Enter short when strong downward momentum is detected
- Use VWAP analysis for additional confirmation

Requirements:
- Tick data ingested into Vegas database (automatically converted to tick bars)
- Example uses SPY tick data from 2024-04-24
"""

from datetime import datetime
from typing import List

import polars as pl

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Context, Signal
from vegas.utils import tabulate_results


class TickBarStrategy(Strategy):
    """Simple momentum strategy using tick bars aggregated from tick data."""
    
    def initialize(self, context: Context):
        """Initialize the strategy with parameters."""
        context.lookback_bars = 10  # Number of tick bars to look back
        context.momentum_threshold = 0.002  # Require .2% price momentum to trade
        context.symbols = ["SPY"]
        context.position_size = 100  # Number of shares to trade
        
        # Store reference to logger
        context.logger = context.engine._logger if hasattr(context, 'engine') else None
    
    def handle_data(self, context: Context, data) -> List[Signal]:
        """Generate trading signals based on tick bar momentum analysis.
        
        This method is called for each timestamp during the backtest.
        It analyzes recent tick bars and generates buy/sell signals based on price momentum.
        """
        signals = []
        current_time = context.current_ts
        
        # Get historical tick bar data from data portal for momentum analysis
        hist_data = data.history(
            assets=["SPY"], 
            bar_count=context.lookback_bars,  # Get recent tick bars for analysis
            frequency="tick:1000"  # Request tick bars (1000 ticks per bar)
        )
            
        first_price = hist_data.row(0, named=True)["close"]  # First bar's close
        last_price = hist_data.row(-1, named=True)["close"]  # Last bar's close
        
        # Calculate price momentum as percentage change
        price_momentum = (last_price - first_price) / first_price
        
        # Calculate volume-weighted average price (VWAP) for the period
        recent_bars_with_vwap = hist_data.with_columns([(pl.col("close") * pl.col("volume")).alias("dollar_volume")])
            
        total_volume = recent_bars_with_vwap.select(pl.col("volume").sum()).row(0)[0]
        total_dollar_volume = recent_bars_with_vwap.select(pl.col("dollar_volume").sum()).row(0)[0]
        
        if total_volume > 0:
            vwap = total_dollar_volume / total_volume
            current_price = last_price
            price_vs_vwap = (current_price - vwap) / vwap
        
        # Get current positions
        current_position = 0
        positions = context.portfolio.get_positions()
        for pos in positions:
            if pos.symbol == "SPY":
                current_position = pos.quantity
                break
        
        if price_momentum >= context.momentum_threshold and current_position <= 0:
            # Strong upward momentum - go long
            quantity = context.position_size
            if current_position < 0:
                quantity += abs(current_position)  # Cover short and go long
                
            signals.append(Signal(
                symbol="SPY",
                quantity=quantity  # positive quantity = buy
            ))
            context.last_trade_time = current_time
                
        elif price_momentum <= -context.momentum_threshold and current_position >= 0:
            # Strong downward momentum - go short
            quantity = context.position_size
            if current_position > 0:
                quantity += current_position  # Close long and go short
                
            signals.append(Signal(
                symbol="SPY", 
                quantity=-quantity  # negative quantity = sell
            ))
            context.last_trade_time = current_time
        
        return signals


def main():
    """Main function to run the tick bar backtest example."""
    
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
        
    # Create the backtest engine 
    engine = BacktestEngine()
    
    start_time = datetime(2025, 6, 1)
    end_time = datetime(2025, 6, 30)
    
    # Create and initialize strategy
    strategy = TickBarStrategy()
    
    # Run the backtest using Vegas engine with tick bars
    results = engine.run(
        start=start_time,
        end=end_time,
        strategy=strategy,
        initial_capital=100000.0,
        frequency="tick:1000",  # Use tick bars of 1000 ticks each 
        data_type="tick"  # Use tick data transformed into tick bars
    )

    print(tabulate_results(results.stats))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Weekly SPY Trading Strategy Example

This script implements a simple strategy that:
1. Buys 100 shares of SPY on Monday of each week at 4am EST
2. Sells all SPY shares on Friday of each week at 7pm EST
"""
import polars as pl
from vegas.strategy import Strategy, Signal, Context

class WeeklySpyTrader(Strategy):
    """Simple strategy that buys SPY on Monday and sells on Friday."""
    
    def initialize(self, context: Context) -> None:
        """Initialize the strategy.
        
        Args:
            context: Strategy context
        """
        context.symbols = ["SPY"]  # Trading only SPY
        context.shares = 100    # Buy/sell 100 shares each time
        context.currently_holding = False
    
    def before_trading_start(self, context: Context, data: pl.DataFrame) -> None:
        """Set the current date in the context.
        
        Args:
            context: Strategy context
            data: Market data for current day
        """
        if not data.is_empty():
            context.current_date = data.select('timestamp').row(0)[0].date()
            
    def handle_data(self, context: Context, data: pl.DataFrame) -> list[Signal]:
        """Process market data and generate trading signals.
        
        Args:
            context: Strategy context
            data: Market data for current time step
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Filter for SPY data
        spy_data = data.filter(pl.col('symbol') == context.symbols[0])
        
        if spy_data.is_empty():
            return signals
        
        # Get current timestamp
        timestamp = spy_data.select('timestamp').row(0)[0]
        
        # Check if it's Monday (weekday=0) and 4am EST
        if timestamp.weekday() == 0 and timestamp.hour == 4 and not context.currently_holding:
            # Buy on Monday
            signals.append(Signal(
                symbol=context.symbols[0],
                action="buy",
                quantity=context.shares,
                price=None  # Use market price
            ))
            context.currently_holding = True
            
        # Check if it's Friday (weekday=4) and 7pm EST
        elif timestamp.weekday() == 4 and timestamp.hour == 19 and context.currently_holding:
            # Sell on Friday
            signals.append(Signal(
                symbol=context.symbols[0],
                action="sell",
                quantity=context.shares,
                price=None  # Use market price
            ))
            context.currently_holding = False
            
        return signals

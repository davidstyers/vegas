#!/usr/bin/env python3
"""Weekly SPY Trading Strategy Example

This script implements a simple strategy that:
1. Buys 100 shares of SPY on Monday of each week
2. Sells all SPY shares on Friday of each week
"""
import pandas as pd
from vegas.strategy import Strategy, Signal, Context

class WeeklySpyTrader(Strategy):
    """Simple strategy that buys SPY on Monday and sells on Friday."""
    
    def initialize(self, context: Context) -> None:
        """Initialize the strategy.
        
        Args:
            context: Strategy context
        """
        context.symbol = "SPY"  # Trading only SPY
        context.shares = 100    # Buy/sell 100 shares each time
        context.currently_holding = False
    
    def before_trading_start(self, context: Context, data: pd.DataFrame) -> None:
        """Set the current date in the context.
        
        Args:
            context: Strategy context
            data: Market data for current day
        """
        if not data.empty:
            context.current_date = data['timestamp'].iloc[0].date()
            
    def handle_data(self, context: Context, data: pd.DataFrame) -> list[Signal]:
        """Process market data and generate trading signals.
        
        Args:
            context: Strategy context
            data: Market data for current time step
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Filter for SPY data
        spy_data = data[data['symbol'] == context.symbol]
        
        if spy_data.empty:
            return signals
        
        # Get current timestamp
        timestamp = spy_data['timestamp'].iloc[0]
        
        # Check if it's Monday (weekday=0)
        if timestamp.weekday() == 0 and not context.currently_holding:
            # Buy on Monday
            signals.append(Signal(
                symbol=context.symbol,
                action="buy",
                quantity=context.shares,
                price=None  # Use market price
            ))
            context.currently_holding = True
            
        # Check if it's Friday (weekday=4)
        elif timestamp.weekday() == 4 and context.currently_holding:
            # Sell on Friday
            signals.append(Signal(
                symbol=context.symbol,
                action="sell",
                quantity=context.shares,
                price=None  # Use market price
            ))
            context.currently_holding = False
            
        return signals

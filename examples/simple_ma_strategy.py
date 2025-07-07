#!/usr/bin/env python3
"""Simple Moving Average Strategy Example

This is a basic example strategy that can be run with the Vegas CLI.
"""

import pandas as pd
import numpy as np
from vegas.strategy import Strategy, Context


class SimpleMovingAverageStrategy(Strategy):
    """Simple Moving Average Crossover Strategy.
    
    This strategy buys when the price crosses above a moving average
    and sells when it crosses below.
    """
    
    def initialize(self, context: Context) -> None:
        """Initialize strategy parameters.
        
        Args:
            context: Strategy context
        """
        # Strategy parameters
        context.symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN']  # Example symbols
        context.ma_window = 20  # Moving average window
        context.position_size = 100  # Number of shares to trade
    
    def generate_signals_vectorized(self, context: Context, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals in a vectorized manner.
        
        Args:
            context: Strategy context
            data: DataFrame with all market data
            
        Returns:
            DataFrame with signals
        """
        # Filter data for our symbols of interest
        if context.symbols:
            data = data[data['symbol'].isin(context.symbols)]
        
        if data.empty:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        # Create an empty list to store signals
        signals = []
        
        # Process each symbol separately
        for symbol in data['symbol'].unique():
            # Get data for this symbol and sort by timestamp
            symbol_data = data[data['symbol'] == symbol].sort_values('timestamp')
            
            # Need enough data for the moving average
            if len(symbol_data) <= context.ma_window:
                continue
            
            # Calculate the moving average
            symbol_data['ma'] = symbol_data['close'].rolling(window=context.ma_window).mean()
            
            # Create shifted data to detect crossovers
            symbol_data['prev_close'] = symbol_data['close'].shift(1)
            symbol_data['prev_ma'] = symbol_data['ma'].shift(1)
            
            # Skip the beginning where we don't have enough data
            symbol_data = symbol_data.dropna()
            
            # Buy signal: price crosses above MA
            symbol_data['buy_signal'] = (symbol_data['prev_close'] < symbol_data['prev_ma']) & \
                                       (symbol_data['close'] > symbol_data['ma'])
            
            # Sell signal: price crosses below MA
            symbol_data['sell_signal'] = (symbol_data['prev_close'] > symbol_data['prev_ma']) & \
                                        (symbol_data['close'] < symbol_data['ma'])
            
            # Generate buy signals
            buy_signals = symbol_data[symbol_data['buy_signal']].apply(
                lambda row: {
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': context.position_size,
                    'price': None  # Market order
                }, axis=1
            ).tolist()
            
            # Generate sell signals
            sell_signals = symbol_data[symbol_data['sell_signal']].apply(
                lambda row: {
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': context.position_size,
                    'price': None  # Market order
                }, axis=1
            ).tolist()
            
            signals.extend(buy_signals)
            signals.extend(sell_signals)
        
        if not signals:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
            
        return pd.DataFrame(signals)
    
    def analyze(self, context: Context, results: dict) -> None:
        """Analyze backtest results.
        
        Args:
            context: Strategy context
            results: Backtest results
        """
        pass
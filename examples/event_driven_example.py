#!/usr/bin/env python3
"""Example of an event-driven strategy using Vegas backtesting engine.

This example shows how to implement an event-driven strategy
that uses before_trading_start, on_market_open, and on_market_close events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import Vegas modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Signal, Context
from vegas.data import DataLayer


class EventDrivenMAStrategy(Strategy):
    """Simple moving average crossover strategy using event-driven approach.
    
    This strategy computes signals at market open based on moving average
    crossovers calculated during before_trading_start.
    """
    
    def initialize(self, context):
        """Initialize strategy parameters."""
        # Strategy parameters
        context.symbols = ['AAPL', 'MSFT', 'GOOG']
        context.short_window = 10  # Short moving average window
        context.long_window = 30   # Long moving average window
        context.signals = {}       # Store signals for each symbol
        
        # This strategy explicitly opts into event-driven mode
        # (though it would be detected automatically due to event methods)
        self.is_event_driven = True
    
    def before_trading_start(self, context, data):
        """Run before market open to compute signals for the day."""
        print(f"Before trading start: {data['timestamp'].iloc[0] if not data.empty else None}")
        
        context.signals = {}
        
        # Get historical data for each symbol to compute signals
        for symbol in context.symbols:
            # In a real implementation, we would use a proper lookback
            # Here we just use the available data
            symbol_data = data[data['symbol'] == symbol]
            
            if len(symbol_data) < context.long_window:
                print(f"Not enough data for {symbol}")
                continue
                
            # Calculate moving averages
            closes = symbol_data['close']
            short_ma = closes.rolling(window=context.short_window).mean()
            long_ma = closes.rolling(window=context.long_window).mean()
            
            # Check for crossover
            if len(short_ma) > 1 and len(long_ma) > 1:
                current_short = short_ma.iloc[-1]
                current_long = long_ma.iloc[-1]
                prev_short = short_ma.iloc[-2] if len(short_ma) > 2 else None
                prev_long = long_ma.iloc[-2] if len(long_ma) > 2 else None
                
                # Buy signal: short MA crosses above long MA
                if prev_short and prev_long:
                    if prev_short <= prev_long and current_short > current_long:
                        context.signals[symbol] = 'buy'
                        print(f"BUY signal for {symbol}: short_ma={current_short:.2f}, long_ma={current_long:.2f}")
                        
                    # Sell signal: short MA crosses below long MA
                    elif prev_short >= prev_long and current_short < current_long:
                        context.signals[symbol] = 'sell'
                        print(f"SELL signal for {symbol}: short_ma={current_short:.2f}, long_ma={current_long:.2f}")
    
    def on_market_open(self, context, data, portfolio):
        """Execute signals at market open."""
        print(f"Market open: {data['timestamp'].iloc[0] if not data.empty else None}")
        
        # Nothing to do if no signals
        if not context.signals:
            return
        
        # Current positions
        positions = {}
        if portfolio:
            for _, pos in portfolio.positions.items():
                positions[pos['symbol']] = pos['quantity']
        
        # Execute signals
        for symbol, signal in context.signals.items():
            if signal == 'buy' and symbol not in positions:
                # Buy 10 shares
                print(f"Executing BUY for {symbol}")
                # We'll return the signal via handle_data
            elif signal == 'sell' and symbol in positions:
                # Sell all shares
                print(f"Executing SELL for {symbol}")
                # We'll return the signal via handle_data
    
    def on_market_close(self, context, data, portfolio):
        """Optional processing at market close."""
        print(f"Market close: {data['timestamp'].iloc[0] if not data.empty else None}")
        
        # Print current portfolio status
        print("\nPortfolio status at close:")
        print(f"Cash: ${portfolio.current_cash:.2f}")
        print(f"Equity: ${portfolio.current_equity:.2f}")
        print("Positions:")
        for symbol, quantity in portfolio.positions.items():
            print(f"  {symbol}: {quantity} shares")
    
    def handle_data(self, context, data):
        """Process data and generate signals."""
        signals = []
        
        # Execute signals from on_market_open
        for symbol, signal_type in context.signals.items():
            if symbol in data['symbol'].values:
                price = data[data['symbol'] == symbol]['close'].iloc[0]
                
                if signal_type == 'buy':
                    signals.append(Signal(
                        symbol=symbol,
                        action='buy',
                        quantity=10,  # Buy 10 shares
                        price=price
                    ))
                elif signal_type == 'sell':
                    # Sell all shares
                    signals.append(Signal(
                        symbol=symbol,
                        action='sell',
                        quantity=10,  # For simplicity, we're just selling 10 shares
                        price=price
                    ))
        
        return signals
    
    def analyze(self, context, results):
        """Analyze backtest results."""
        print("\nBacktest completed.")
        print(f"Total return: {results['stats']['total_return_pct']:.2f}%")
        print(f"Number of trades: {results['stats']['num_trades']}")
        
        # Plot equity curve if data available
        if 'equity_curve' in results and not results['equity_curve'].empty:
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve']['timestamp'], results['equity_curve']['equity'])
            plt.title('Portfolio Equity')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('event_driven_ma_equity.png')
            plt.close()
            print("Equity curve saved to event_driven_ma_equity.png")


if __name__ == "__main__":
    # Initialize backtest engine
    engine = BacktestEngine()
    
    # Load test data (replace with your own data path)
    engine.load_data(directory="tests/db")
    
    # Set up the strategy
    strategy = EventDrivenMAStrategy()
    
    # Run backtest
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)
    results = engine.run(
        start=start_date,
        end=end_date,
        strategy=strategy,
        initial_capital=100000.0,
        event_driven=True  # Explicitly request event-driven mode (optional)
    )
    
    print("\nBacktest results:")
    print(f"Final portfolio value: ${results['equity_curve']['equity'].iloc[-1]:.2f}")
    print(f"Total return: {results['stats']['total_return_pct']:.2f}%")
    print(f"Total trades: {results['stats']['num_trades']}") 
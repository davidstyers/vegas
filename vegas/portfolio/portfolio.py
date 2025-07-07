"""Portfolio layer for the Vegas backtesting engine.

This module provides a minimal portfolio tracking system optimized for vectorized backtesting.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np


class Portfolio:
    """Minimal vectorized portfolio tracking for the Vegas backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize the portfolio.
        
        Args:
            initial_capital: Initial portfolio cash balance
        """
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.position_values = {}  # symbol -> market value
        self.current_equity = initial_capital
        
        # For historical tracking
        self.equity_history = []  # List of {timestamp, equity, cash}
        self.position_history = []  # List of {timestamp, symbol, quantity, value}
        self.transaction_history = []  # List of {timestamp, symbol, quantity, price, commission}
        
    def update_from_transactions(self, timestamp: datetime, transactions: pd.DataFrame, market_data: pd.DataFrame) -> None:
        """Update portfolio based on executed transactions and current market data.
        
        Args:
            timestamp: Current timestamp
            transactions: DataFrame with executed transactions (columns: symbol, quantity, price, commission)
            market_data: DataFrame with current market data (must include columns: symbol, close)
        """
        # Process transactions
        if not transactions.empty:
            for _, txn in transactions.iterrows():
                symbol = txn['symbol']
                quantity = txn['quantity']
                price = txn['price']
                commission = txn.get('commission', 0.0)
                
                # Update cash
                self.current_cash -= (quantity * price + commission)
                
                # Update positions
                if symbol not in self.positions:
                    self.positions[symbol] = 0
                self.positions[symbol] += quantity
                
                # Remove positions with zero quantity
                if abs(self.positions[symbol]) < 1e-6:
                    del self.positions[symbol]
                
                # Record transaction
                self.transaction_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'commission': commission
                })
        
        # Update position values based on current market data
        self.position_values = {}
        total_position_value = 0.0
        
        if self.positions:
            # Create a lookup table for current prices
            price_lookup = market_data.set_index('symbol')['close'].to_dict()
            
            for symbol, quantity in self.positions.items():
                if symbol in price_lookup:
                    price = price_lookup[symbol]
                    value = quantity * price
                    self.position_values[symbol] = value
                    total_position_value += value
                    
                    # Record position
                    self.position_history.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'quantity': quantity,
                        'value': value
                    })
        
        # Update equity
        self.current_equity = self.current_cash + total_position_value
        
        # Record equity history
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': self.current_equity,
            'cash': self.current_cash
        })
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        position_value = sum(self.position_values.values())
        return self.current_cash + position_value
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get portfolio equity curve as a DataFrame.
        
        Returns:
            DataFrame with timestamp and equity values
        """
        if not self.equity_history:
            return pd.DataFrame(columns=['timestamp', 'equity', 'cash'])
            
        return pd.DataFrame(self.equity_history)
    
    def get_returns(self) -> pd.DataFrame:
        """Get portfolio returns as a DataFrame.
        
        Returns:
            DataFrame with timestamp and return values
        """
        if not self.equity_history or len(self.equity_history) < 2:
            return pd.DataFrame(columns=['timestamp', 'return'])
            
        equity_df = pd.DataFrame(self.equity_history)
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Fill the first NaN return with 0
        equity_df['return'] = equity_df['return'].fillna(0)
        
        return equity_df[['timestamp', 'return']]
    
    def get_transactions(self) -> pd.DataFrame:
        """Get transaction history as a DataFrame.
        
        Returns:
            DataFrame with transaction details
        """
        if not self.transaction_history:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'quantity', 'price', 'commission'])
            
        return pd.DataFrame(self.transaction_history)
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions as a DataFrame.
        
        Returns:
            DataFrame with position details
        """
        positions = []
        for symbol, quantity in self.positions.items():
            value = self.position_values.get(symbol, 0.0)
            positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'value': value
            })
        
        if not positions:
            return pd.DataFrame(columns=['symbol', 'quantity', 'value'])
            
        return pd.DataFrame(positions)
    
    def get_positions_history(self) -> Dict[datetime, Dict[str, Dict[str, float]]]:
        """Get historical positions data.
        
        Returns:
            Dictionary with timestamp -> symbol -> position details
        """
        positions_by_time = {}
        
        for pos in self.position_history:
            timestamp = pos['timestamp']
            symbol = pos['symbol']
            
            if timestamp not in positions_by_time:
                positions_by_time[timestamp] = {}
                
            positions_by_time[timestamp][symbol] = {
                'quantity': pos['quantity'],
                'value': pos['value']
            }
            
        return positions_by_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate basic performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.equity_history:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0
            }
        
        # Calculate metrics
        initial_equity = self.initial_capital
        final_equity = self.current_equity
        
        total_return = final_equity - initial_equity
        total_return_pct = (total_return / initial_equity) if initial_equity > 0 else 0.0
        
        # Count trades
        num_trades = len(self.transaction_history)
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct * 100.0,  # Convert to percentage
            'num_trades': num_trades
        } 
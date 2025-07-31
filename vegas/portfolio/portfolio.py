"""Portfolio layer for the Vegas backtesting engine.

This module provides a portfolio tracking system for event-driven backtesting.
"""

from typing import Dict, Any
from datetime import datetime
import logging
import pandas as pd


class Position:
    """A class representing a portfolio position."""
    
    def __init__(self, symbol, quantity, value=0.0):
        """Initialize a position.
        
        Args:
            symbol: Symbol of the security
            quantity: Number of shares/units
            value: Current market value
        """
        self.symbol = symbol
        self.quantity = quantity
        self.value = value
        
    def __str__(self):
        return f"Position({self.symbol}, {self.quantity}, ${self.value:.2f})"
    
    def __repr__(self):
        return self.__str__()

class Portfolio:
    """Portfolio tracking system for the Vegas event-driven backtesting engine."""
    
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
        
        # Setup logging
        self._logger = logging.getLogger('vegas.portfolio')
    
    def update_from_transactions(self, timestamp, transactions, market_data):
        """Update portfolio based on executed transactions and current market data.
        
        Args:
            timestamp: Current timestamp
            transactions: DataFrame with executed transactions (columns: symbol, quantity, price, commission)
            market_data: DataFrame with current market data (must include columns: symbol, close)
        """
        # Process transactions
        if not transactions.is_empty():
            for txn in transactions.to_dicts():
                symbol = txn['symbol']
                quantity = txn['quantity']
                price = txn['price']
                commission = txn.get('commission', 0.0)
                self._logger.info(f"{timestamp[0].strftime("%Y-%m-%d %H:%M:%S")}: {quantity} of {symbol} at {price}")

                
                # For sell transactions, validate we have enough shares
                if quantity < 0:
                    current_position = self.positions.get(symbol, 0)
                    if abs(quantity) > current_position:
                        original_quantity = quantity
                        # Cannot sell more than we own (prevent short selling)
                        quantity = -current_position
                        if abs(quantity) < 1e-6:
                            self._logger.warning(
                                f"Skipping sell of {abs(original_quantity)} shares of {symbol} - "
                                f"insufficient position (current: {current_position})"
                            )
                            continue  # Skip if adjusted quantity is effectively zero
                        self._logger.warning(
                            f"Adjusted sell from {abs(original_quantity)} to {abs(quantity)} shares of "
                            f"{symbol} due to insufficient position"
                        )
                
                # For buy transactions, validate we have enough cash
                if quantity > 0:
                    cost = quantity * price + commission
                    if cost > self.current_cash:
                        original_quantity = quantity
                        # Scale back to what we can afford
                        max_quantity = max(0, (self.current_cash - commission) / price)
                        quantity = max_quantity
                        if quantity < 1e-6:
                            self._logger.warning(
                                f"Skipping buy of {original_quantity} shares of {symbol} - "
                                f"insufficient cash (needed: ${cost:.2f}, available: ${self.current_cash:.2f})"
                            )
                            continue  # Skip if adjusted quantity is effectively zero
                        # Recalculate commission with new quantity
                        commission = txn.get('commission', 0.0) * (quantity / txn['quantity'])
                        self._logger.warning(
                            f"Adjusted buy from {original_quantity} to {quantity} shares of "
                            f"{symbol} due to insufficient cash"
                        )
                
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
            price_lookup = dict(zip(market_data['symbol'], market_data['close']))
            
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
    
    def get_portfolio_value(self):
        """Get the current portfolio value.
        
        Returns:
            Total portfolio value (cash + positions)
        """
        return self.current_equity
    
    def get_equity_curve(self):
        """Get the portfolio equity curve.
        
        Returns:
            DataFrame with columns: timestamp, equity, cash
        """
        if not self.equity_history:
            return pd.DataFrame(columns=['timestamp', 'equity', 'cash'])
        return pd.DataFrame(self.equity_history)
    
    def get_returns(self):
        """Get portfolio returns.
        
        Returns:
            DataFrame with portfolio returns
        """
        equity_curve = self.get_equity_curve()
        if len(equity_curve) <= 1:
            return pd.DataFrame(columns=['timestamp', 'return', 'cumulative_return'])
        
        # Calculate returns with explicit fill_method parameter to avoid deprecation warning
        equity_curve['return'] = equity_curve['equity'].pct_change(fill_method=None).fillna(0)
        equity_curve['cumulative_return'] = (1 + equity_curve['return']).cumprod() - 1
        
        return equity_curve[['timestamp', 'return', 'cumulative_return']]
    
    def get_transactions(self):
        """Get all transactions.
        
        Returns:
            DataFrame with transaction history
        """
        if not self.transaction_history:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'quantity', 'price', 'commission'])
        return pd.DataFrame(self.transaction_history)
    
    def get_positions(self):
        """Get current portfolio positions as Position objects.
        
        Returns:
            List of Position objects
        """
        positions_list = []
        for symbol, quantity in self.positions.items():
            if abs(quantity) > 1e-6:  # Only include non-zero positions
                value = self.position_values.get(symbol, 0.0)
                positions_list.append(Position(symbol, quantity, value))
        
        return positions_list
    
    def get_positions_dataframe(self):
        """Get current portfolio positions as a DataFrame.
        
        Returns:
            DataFrame with current positions
        """
        positions_data = []
        for symbol, quantity in self.positions.items():
            value = self.position_values.get(symbol, 0.0)
            positions_data.append({
                'symbol': symbol,
                'quantity': quantity,
                'value': value,
                'weight': value / self.current_equity if self.current_equity > 0 else 0.0
            })
            
        if not positions_data:
            return pd.DataFrame(columns=['symbol', 'quantity', 'value', 'weight'])
        return pd.DataFrame(positions_data)
    
    def get_positions_history(self):
        """Get the history of portfolio positions.
        
        Returns:
            Dict with timestamp-keyed position snapshots
        """
        position_history = {}
        
        # Group position history by timestamp
        df = pd.DataFrame(self.position_history)
        if not df.empty:
            # Convert to dictionary format: timestamp -> {symbol -> {quantity, value}}
            for timestamp, group in df.groupby('timestamp'):
                symbol_dict = {}
                for _, row in group.iterrows():
                    symbol_dict[row['symbol']] = {
                        'quantity': row['quantity'],
                        'value': row['value']
                    }
                position_history[timestamp] = symbol_dict
                
        return position_history
    
    def get_stats(self):
        """Calculate performance statistics.
        
        Returns:
            Dictionary with calculated statistics
        """
        stats = {}
        
        # Basic portfolio stats
        stats['initial_capital'] = self.initial_capital
        stats['final_value'] = self.current_equity
        stats['cash'] = self.current_cash
        stats['total_return'] = self.current_equity - self.initial_capital
        stats['total_return_pct'] = (self.current_equity / self.initial_capital - 1) * 100
        
        # Transaction stats
        transactions_df = self.get_transactions()
        stats['num_trades'] = len(transactions_df)
        
        # Get equity curve for more sophisticated stats
        equity_df = self.get_equity_curve()
        returns_df = self.get_returns()
        
        if len(equity_df) > 1:
            # Calculate drawdown
            equity_df['previous_peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['previous_peak']) / equity_df['previous_peak'] * 100
            stats['max_drawdown'] = abs(equity_df['drawdown'].min())
            stats['max_drawdown_pct'] = abs(equity_df['drawdown'].min())
            
            # Calculate Sharpe ratio (assuming daily data and risk-free rate of 0)
            if len(returns_df) > 1:
                daily_returns = returns_df['return'].values
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    stats['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
                    stats['annual_return_pct'] = ((1 + daily_returns.mean()) ** 252 - 1) * 100
                else:
                    stats['sharpe_ratio'] = 0.0
                    stats['annual_return_pct'] = 0.0
        
        return stats 
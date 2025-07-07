"""Backtest Engine for the Vegas backtesting engine.

This module provides a minimal, vectorized backtesting engine.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import time
import logging
import os

from vegas.data import DataLayer
from vegas.strategy import Strategy, Context, Signal
from vegas.portfolio import Portfolio


class BacktestEngine:
    """Minimal vectorized backtesting engine for the Vegas backtesting system."""
    
    def __init__(self, data_dir: str = "db"):
        """Initialize the backtest engine.
        
        Args:
            data_dir: Directory for storing data files
        """
        self._logger = self._setup_logger()
        self._logger.info(f"Initializing BacktestEngine with data directory: {data_dir}")
        
        self.data_layer = DataLayer(data_dir)
        self.strategy = None
        self.portfolio = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the backtest engine."""
        logger = logging.getLogger('vegas.engine')
        logger.setLevel(logging.INFO)
        
        # Create console handler if not already exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
        
    def load_data(self, file_path: str = None, directory: str = None, file_pattern: str = None, 
                 max_files: int = None) -> None:
        """Load market data for backtesting.
        
        Args:
            file_path: Optional path to a single CSV file
            directory: Optional directory containing multiple data files
            file_pattern: Optional pattern for matching multiple files
            max_files: Optional limit on number of files to load
        """
        if file_path:
            self._logger.info(f"Loading data from single file: {file_path}")
            self.data_layer.load_data(file_path)
        elif directory or file_pattern:
            self._logger.info(f"Loading data from multiple files")
            self.data_layer.load_multiple_files(
                directory=directory,
                file_pattern=file_pattern,
                max_files=max_files
            )
        elif not self.data_layer.is_initialized():
            raise ValueError(
                "No data source specified and no data is currently loaded. "
                "Either provide a file_path, directory, or ensure data is already loaded."
            )
        else:
            self._logger.info("Using already loaded data")
            
        # Log information about the loaded data
        data_info = self.data_layer.get_data_info()
        self._logger.info(f"Available data: {data_info['row_count']} rows, "
                        f"{data_info['symbol_count']} symbols, "
                        f"from {data_info['start_date']} to {data_info['end_date']}")
    
    def run(self, start: datetime, end: datetime, strategy: Strategy,
           initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Run a vectorized backtest.
        
        Args:
            start: Start date for the backtest
            end: End date for the backtest
            strategy: Strategy to run
            initial_capital: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        self._logger.info(f"Starting backtest from {start.date()} to {end.date()}")
        
        # Initialize strategy and portfolio
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital=initial_capital)
        
        # Get context and initialize strategy
        context = self.strategy.context
        self.strategy.initialize(context)
        
        start_time = time.time()
        
        # Get data for the backtest period
        self._logger.info("Loading market data for backtest period")
        market_data = self.data_layer.get_data_for_backtest(start, end)
        
        if market_data.empty:
            self._logger.warning("No data available for the specified period")
            return self._create_empty_results()
        
        # Execute vectorized backtest
        self._logger.info("Executing vectorized backtest")
        results = self._run_vectorized_backtest(market_data, context)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self._logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        # Get final statistics
        stats = self.portfolio.get_stats()
        
        # Create results dictionary
        results_dict = {
            'stats': stats,
            'equity_curve': self.portfolio.get_equity_curve(),
            'transactions': self.portfolio.get_transactions(),
            'positions': self.portfolio.get_positions(),
            'execution_time': execution_time,
            'success': True
        }
        
        # Allow strategy to analyze results
        self.strategy.analyze(context, results_dict)
        
        return results_dict
    
    def _run_vectorized_backtest(self, market_data: pd.DataFrame, context: Context) -> Dict[str, Any]:
        """Run the vectorized backtest algorithm.
        
        Args:
            market_data: DataFrame with market data for the backtest period
            context: Strategy context
            
        Returns:
            Dictionary with backtest results
        """
        # Step 1: Generate signals using the vectorized strategy method
        self._logger.info("Generating trading signals")
        signals_df = self.strategy.generate_signals_vectorized(context, market_data)
        
        if signals_df.empty:
            self._logger.warning("No signals generated by the strategy")
            return {}
        
        # Step 2: Execute signals and update portfolio
        self._logger.info("Executing signals and updating portfolio")
        
        # Sort data by timestamp for sequential processing
        market_data = market_data.sort_values('timestamp')
        signals_df = signals_df.sort_values('timestamp')
        
        # Get unique timestamps
        timestamps = market_data['timestamp'].unique()
        
        # Process data timestamp by timestamp
        for timestamp in timestamps:
            # Get data for current timestamp
            current_data = market_data[market_data['timestamp'] == timestamp]
            
            # Get signals for current timestamp
            current_signals = signals_df[signals_df['timestamp'] == timestamp]
            
            # Execute signals and create transactions
            if not current_signals.empty:
                # Convert signals to transactions (in real implementation, this would include 
                # slippage, commission, and execution logic)
                transactions = self._create_transactions_from_signals(current_signals, current_data)
                
                # Update portfolio with transactions
                if not transactions.empty:
                    self.portfolio.update_from_transactions(timestamp, transactions, current_data)
            else:
                # Update portfolio with current market data only (no transactions)
                self.portfolio.update_from_transactions(timestamp, pd.DataFrame(), current_data)
        
        return {}
    
    def _create_transactions_from_signals(self, signals: pd.DataFrame, 
                                         market_data: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to transactions.
        
        In a real implementation, this would include slippage, commission, and execution logic.
        
        Args:
            signals: DataFrame with signals
            market_data: DataFrame with current market data
            
        Returns:
            DataFrame with transactions
        """
        transactions = []
        
        # Create a lookup for current prices
        price_lookup = market_data.set_index('symbol')['close'].to_dict()
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            
            # Skip if symbol not in current market data
            if symbol not in price_lookup:
                continue
                
            # Get execution price (with optional slippage model)
            price = price_lookup[symbol]
            
            # Convert action to quantity (positive for buy, negative for sell)
            if action.lower() == 'sell':
                quantity = -abs(quantity)
                
            # Add a simplified commission (could be enhanced with a commission model)
            commission = abs(quantity * price * 0.001)  # 0.1% commission
            
            transactions.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission
            })
        
        return pd.DataFrame(transactions) if transactions else pd.DataFrame()
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results dictionary for when no data is available."""
        return {
            'stats': {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0
            },
            'equity_curve': pd.DataFrame(),
            'transactions': pd.DataFrame(),
            'positions': pd.DataFrame(),
            'execution_time': 0.0,
            'success': False
        } 
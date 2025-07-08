"""Backtest Engine for the Vegas backtesting engine.

This module provides a minimal, vectorized backtesting engine.
"""

from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime
import pandas as pd
import time
import logging

from vegas.data import DataLayer
from vegas.strategy import Strategy, Context
from vegas.portfolio import Portfolio

# Import event engine if Cython is available
try:
    from vegas.engine.event_engine import EventDrivenEngine, generate_events
    CYTHON_AVAILABLE = True
except ImportError:
    # Fallback to Python implementations if Cython fails
    CYTHON_AVAILABLE = False
    from vegas.engine.event_engine_py import EventDrivenEngine, generate_events


class BacktestEngine:
    """Backtesting engine for the Vegas backtesting system.
    
    This engine supports both vectorized and event-driven backtests.
    Vectorized mode is used by default for performance, but event-driven
    mode is automatically selected when a strategy implements event methods
    or when explicitly requested.
    """
    
    def __init__(self, data_dir: str = "db"):
        """Initialize the backtest engine.
        
        Args:
            data_dir: Directory for storing data files
        """
        self._logger = logging.getLogger('vegas.engine')
        self._logger.setLevel(logging.INFO)
        
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
            
        self._logger.info(f"Initializing BacktestEngine with data directory: {data_dir}")
        
        self.data_layer = DataLayer(data_dir)
        self.strategy = None
        self.portfolio = None
    
    def load_data(self, file_path: str = None, directory: str = None, 
                 file_pattern: str = None, max_files: int = None) -> None:
        """Load market data for backtesting.
        
        Args:
            file_path: Optional path to a single CSV file
            directory: Optional directory containing multiple data files
            file_pattern: Optional pattern for matching multiple files
            max_files: Optional limit on number of files to load
        """
        # Handle loading data based on provided parameters
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
            raise ValueError("No data source specified and no data is currently loaded.")
        else:
            self._logger.info("Using already loaded data")
            
        # Log information about the loaded data
        data_info = self.data_layer.get_data_info()
        self._logger.info(f"Available data: {data_info['row_count']} rows, "
                        f"{data_info['symbol_count']} symbols, "
                        f"from {data_info['start_date']} to {data_info['end_date']}")
    
    def run(self, start: datetime, end: datetime, strategy: Strategy,
           initial_capital: float = 100000.0, event_driven: bool = None) -> Dict[str, Any]:
        """Run a backtest.
        
        Args:
            start: Start date for the backtest
            end: End date for the backtest
            strategy: Strategy to run
            initial_capital: Initial capital
            event_driven: Override to force event-driven mode (True) or vectorized mode (False)
                          If None, the engine will detect the appropriate mode based on the strategy
            
        Returns:
            Dictionary with backtest results
        """
        self._logger.info(f"Starting backtest from {start.date()} to {end.date()}")
        
        # Initialize strategy and portfolio
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital=initial_capital)
        
        # Get context and initialize strategy
        context = self.strategy.context
        context.set_portfolio(self.portfolio)  # Ensure portfolio is accessible
        self.strategy.initialize(context)
        
        start_time = time.time()
        
        # Get data for the backtest period
        self._logger.info("Loading market data for backtest period")
        market_data = self.data_layer.get_data_for_backtest(start, end)
        
        if market_data.empty:
            self._logger.warning("No data available for the specified period")
            return self._create_empty_results()
        
        # Determine whether to use event-driven or vectorized mode
        should_use_event_driven = self._requires_event_driven(strategy)
        
        # Override if explicitly specified
        if event_driven is not None:
            should_use_event_driven = event_driven
            if event_driven:
                self._logger.info("Event-driven mode explicitly requested")
            else:
                self._logger.info("Vectorized mode explicitly requested")
                
        # Execute the appropriate backtest type
        if should_use_event_driven:
            self._logger.info("Executing event-driven backtest")
            results_dict = self._run_event_driven_backtest(start, end, context)
        else:
            self._logger.info("Executing vectorized backtest")
            self._run_vectorized_backtest(market_data, context)
            
            # Create results dictionary for vectorized mode
            results_dict = {
                'stats': self.portfolio.get_stats(),
                'equity_curve': self.portfolio.get_equity_curve(),
                'transactions': self.portfolio.get_transactions(),
                'positions': self.portfolio.get_positions(),
                'success': True
            }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self._logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        # Add execution time to results
        results_dict['execution_time'] = execution_time
        
        # Allow strategy to analyze results
        self.strategy.analyze(context, results_dict)
        
        return results_dict
    
    def _requires_event_driven(self, strategy: Strategy) -> bool:
        """Check if a strategy requires event-driven execution.
        
        A strategy requires event-driven execution if:
        1. It has explicitly set is_event_driven = True
        2. It implements any of the event-driven methods with custom logic
        
        Args:
            strategy: Strategy to check
            
        Returns:
            True if the strategy should use event-driven mode
        """
        # Check explicit flag first
        if hasattr(strategy, 'is_event_driven') and strategy.is_event_driven:
            return True
            
        # Check if any event methods are implemented (not using default implementation)
        event_methods = [
            'before_trading_start',
            'on_market_open',
            'on_market_close',
            'on_bar',
            'on_tick',
            'on_trade'
        ]
        
        strategy_class = strategy.__class__
        base_class = Strategy
        
        for method_name in event_methods:
            # Check if the method is overridden
            if hasattr(strategy_class, method_name):
                strategy_method = getattr(strategy_class, method_name)
                base_method = getattr(base_class, method_name)
                
                # If the method implementation is different from the base class
                if strategy_method.__code__ is not base_method.__code__:
                    return True
                    
        return False
    
    def _run_vectorized_backtest(self, market_data: pd.DataFrame, context: Context) -> None:
        """Run the vectorized backtest algorithm.
        
        Args:
            market_data: DataFrame with market data for the backtest period
            context: Strategy context
        """
        # Generate signals using the vectorized strategy method
        self._logger.info("Generating trading signals")
        
        try:
            signals_df = self.strategy.generate_signals_vectorized(context, market_data)
        except Exception as e:
            self._logger.error(f"Error generating signals: {e}")
            signals_df = pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        
        if signals_df.empty:
            self._logger.warning("No signals generated by the strategy")
            return
        
        # Process data timestamp by timestamp
        self._logger.info("Executing signals and updating portfolio")
        
        # Sort data by timestamp for sequential processing
        market_data = market_data.sort_values('timestamp')
        signals_df = signals_df.sort_values('timestamp')
        
        # Get unique timestamps
        timestamps = market_data['timestamp'].unique()
        
        # Process each timestamp
        for timestamp in timestamps:
            current_data = market_data[market_data['timestamp'] == timestamp]
            current_signals = signals_df[signals_df['timestamp'] == timestamp]
            
            # Execute signals and create transactions
            transactions = pd.DataFrame()
            if not current_signals.empty:
                transactions = self._create_transactions_from_signals(current_signals, current_data)
                
            # Update portfolio with transactions (or empty DataFrame if no transactions)
            self.portfolio.update_from_transactions(timestamp, transactions, current_data)
    
    def _run_event_driven_backtest(self, start: datetime, end: datetime, context: Context) -> Dict[str, Any]:
        """Run the event-driven backtest algorithm.
        
        Args:
            start: Start date for the backtest
            end: End date for the backtest
            context: Strategy context
            
        Returns:
            Dictionary with backtest results
        """
        # Generate events for the backtest period
        self._logger.info("Generating events for event-driven backtest")
        events_df = generate_events(start, end, self.data_layer, debug=False)
        
        if events_df.empty:
            self._logger.warning("No events generated for the backtest period")
            return self._create_empty_results()
            
        # Create event-driven engine
        if CYTHON_AVAILABLE:
            self._logger.info("Using Cython-optimized event engine")
        else:
            self._logger.warning("Cython event engine not available, using Python fallback")
            
        event_engine = EventDrivenEngine(
            self.strategy, 
            self.portfolio, 
            self.data_layer, 
            self._logger
        )
        
        # Run the event-driven backtest
        return event_engine.run_event_driven_backtest(events_df)
    
    def _create_transactions_from_signals(self, signals: pd.DataFrame, 
                                         market_data: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to transactions.
        
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
                
            # Get execution price
            price = signal.get('price') or price_lookup[symbol]
            
            # Convert action to quantity (positive for buy, negative for sell)
            if action.lower() == 'sell':
                quantity = -abs(quantity)
                
            # Add a simplified commission
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
                'num_trades': 0
            },
            'equity_curve': pd.DataFrame(columns=['timestamp', 'equity', 'cash']),
            'transactions': pd.DataFrame(),
            'positions': pd.DataFrame(),
            'execution_time': 0.0,
            'success': False
        } 
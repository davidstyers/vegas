"""Backtest Engine for the Vegas backtesting engine.

This module provides a streamlined backtesting engine.
"""

from datetime import datetime
import logging
import time
import polars as pl

from vegas.data import DataLayer
from vegas.strategy import Strategy, Context, Signal
from vegas.portfolio import Portfolio
from vegas.pipeline.engine import PipelineEngine


class BacktestEngine:
    """Efficient backtesting engine for the Vegas backtesting system.
    
    This engine provides streamlined backtesting capabilities by directly
    processing market data chronologically without a separate events system.
    """
    
    def __init__(self, data_dir: str = "db", timezone: str = "UTC"):
        """Initialize the backtest engine.
        
        Args:
            data_dir: Directory for storing data files
            timezone: Timezone for data timestamps (default: 'UTC')
        """
        self._logger = logging.getLogger('vegas.engine')
        self._logger.info(f"Initializing BacktestEngine with data directory: {data_dir}, timezone: {timezone}")
        
        self.data_layer = DataLayer(data_dir, timezone=timezone)
        self.strategy = None
        self.portfolio = None
        self.timezone = timezone
        
        # Trading hours configuration
        self._market_open_time = "09:30"  # Default market open (US)
        self._market_close_time = "16:00"  # Default market close (US)
        self._market_name = "US"  # Default market name
        self._ignore_extended_hours = False  # By default, use all data
        
        # Add pipeline engine
        self.pipeline_engine = PipelineEngine(self.data_layer)
        self.attached_pipelines = {}
        self._pipeline_results = {}
    
    def set_trading_hours(self, market_name="US", open_time="09:30", close_time="16:00"):
        """Set the regular trading hours for the backtest.
        
        Args:
            market_name: Market name (e.g., 'NASDAQ', 'NYSE', 'LSE')
            open_time: Market open time in 24-hour format (HH:MM)
            close_time: Market close time in 24-hour format (HH:MM)
        """
        self._market_name = market_name
        self._market_open_time = open_time
        self._market_close_time = close_time
        self._logger.info(f"Set trading hours for {market_name}: {open_time} to {close_time}")
        
    def ignore_extended_hours(self, ignore=True):
        """Configure whether to ignore extended hours data.
        
        Args:
            ignore: If True, only use data within regular market hours
        """
        self._ignore_extended_hours = ignore
        status = "ignored" if ignore else "included"
        self._logger.info(f"Extended hours data will be {status}")
    
    def attach_pipeline(self, pipeline, name):
        """Register a pipeline to be computed at the start of each day.
        
        Args:
            pipeline: The pipeline to compute
            name: The name of the pipeline
            
        Returns:
            The pipeline that was attached
        """
        self._logger.info(f"Attaching pipeline '{name}'")
        self.attached_pipelines[name] = pipeline
        return pipeline
    
    def pipeline_output(self, name):
        """Get the results of the pipeline with the given name for the current day.
        
        Args:
            name: Name of the pipeline
            
        Returns:
            DataFrame containing the results of the requested pipeline for the current date
        """
        if name not in self._pipeline_results:
            self._logger.warning(f"No pipeline named '{name}' has been attached or computed")
            # Return empty DataFrame instead of raising an error
            return pl.DataFrame()
        return self._pipeline_results[name]
        
    def _is_regular_market_hours(self, timestamp):
        """Check if a timestamp falls within regular market hours.
        
        Args:
            timestamp: The timestamp to check
            
        Returns:
            True if timestamp is within regular market hours, False otherwise
        """
        # Convert timestamp to datetime with time component
        dt = timestamp
        
        # Parse market hours
        open_hour, open_minute = map(int, self._market_open_time.split(':'))
        close_hour, close_minute = map(int, self._market_close_time.split(':'))
        
        # Convert to comparable values (minutes since midnight)
        ts_minutes = dt.hour * 60 + dt.minute
        open_minutes = open_hour * 60 + open_minute
        close_minutes = close_hour * 60 + close_minute
        
        # Check if timestamp is within market hours
        return open_minutes <= ts_minutes < close_minutes
    
    def load_data(self, file_path=None, directory=None, file_pattern=None, max_files=None):
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
            self.data_layer.load_data(file_path=file_path)
        elif directory:
            self._logger.info(f"Loading data from directory: {directory}")
            self.data_layer.load_data(directory=directory, file_pattern=file_pattern, max_files=max_files)
        elif not self.data_layer.is_initialized():
            raise ValueError("No data source specified and no data is currently loaded.")
        else:
            self._logger.info("Using already loaded data")
            
        # Log information about the loaded data
        data_info = self.data_layer.get_data_info()
        self._logger.info(
            f"Available data: {data_info.get('row_count', 'unknown')} rows, "
            f"{data_info.get('symbol_count', 0)} symbols, "
            f"from {data_info.get('start_date', 'unknown')} to {data_info.get('end_date', 'unknown')}"
        )
    
    def run(self, start, end, strategy, initial_capital=100000.0):
        """Run a backtest.
        
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
        context.set_portfolio(self.portfolio)  # Ensure portfolio is accessible
        context.set_engine(self)  # Set reference to the engine for pipeline access
        self.strategy.initialize(context)
        
        start_time = time.time()
        
        # Get data for the backtest period
        self._logger.info("Loading market data for backtest period")
        
        # Add market hours filtering at database level if requested
        market_hours = None
        if self._ignore_extended_hours:
            market_hours = (self._market_open_time, self._market_close_time)
        market_data = self.data_layer.get_data_for_backtest(start, end, market_hours=market_hours, symbols=["SPY"])
        
        if market_data.is_empty():
            self._logger.warning("No data available for the specified period")
            return self._create_empty_results()
        
        # Run the backtest
        self._logger.info("Executing backtest")
        results_dict = self._run_backtest(market_data, context)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self._logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        # Add execution time to results
        results_dict['execution_time'] = execution_time
        
        # Allow strategy to analyze results
        self.strategy.analyze(context, results_dict)
        
        return results_dict
    
    def _run_backtest(self, market_data, context):
        """Run the backtest algorithm.
        
        Args:
            market_data: DataFrame with market data for the backtest period
            context: Strategy context
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # The filtering for regular trading hours is now done at database level
            # We just need to check if we have any data
            if market_data.is_empty() and self._ignore_extended_hours:
                self._logger.warning("No data points found within regular market hours!")
                return self._create_empty_results()
            
            # Group market data by trading day for daily processing
            market_data = market_data.with_columns(
                pl.col('timestamp').dt.date().alias('date')
            )
            unique_dates = sorted(market_data.select('date').unique().to_series().to_list())
            self._logger.info(f"Processing {len(unique_dates)} trading days")
            
            # Process data day by day
            for date in unique_dates:
                # Clear previous pipeline results
                self._pipeline_results = {}
                
                # Get data for the current day
                daily_data = market_data.filter(pl.col('date') == date)
                day_timestamp = datetime.combine(date, datetime.min.time())
                
                # Compute any attached pipelines for this day
                for name, pipeline in self.attached_pipelines.items():
                    try:
                        # Run pipeline for just this date
                        pipeline_result = self.pipeline_engine.run_pipeline(
                            pipeline, 
                            start_date=day_timestamp,
                            end_date=day_timestamp
                        )
                        if not pipeline_result.empty:
                            # Make results available via pipeline_output
                            self._pipeline_results[name] = pipeline_result
                            self._logger.debug(
                                f"Pipeline '{name}' computed {len(pipeline_result)} results for {date}"
                            )
                        else:
                            self._logger.warning(f"Pipeline '{name}' returned empty results for {date}")
                    except Exception as e:
                        self._logger.error(f"Error computing pipeline '{name}' for {date}: {e}")
                
                # Call before_trading_start at the beginning of each day
                if hasattr(self.strategy, 'before_trading_start'):
                    # Since we've already filtered at the database level,
                    # we can just use the first data point of the day
                    day_start_data = daily_data.sort('timestamp').head(1)
                        
                    self.strategy.before_trading_start(context, day_start_data)
                
                partitioned = daily_data.sort('timestamp').partition_by('timestamp', as_dict=True)
                sorted_timestamps = sorted(partitioned.keys())
                
                # Process each timestamp chronologically
                for timestamp in sorted_timestamps:
                    timestamp_data = partitioned[timestamp]
                    # Update context with current timestamp
                    context.current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                    
                    # Call handle_data for each timestamp to generate trading signals
                    if hasattr(self.strategy, 'handle_data'):
                        signals = self.strategy.handle_data(context, timestamp_data)
                        
                        if signals:
                            # Process signals and create transactions
                            transactions = self._create_transactions_from_signals(signals, timestamp_data)
                            
                            # Update portfolio with transactions
                            if len(transactions) > 0:
                                self.portfolio.update_from_transactions(
                                    timestamp, 
                                    transactions,
                                    timestamp_data
                                )
                    
                    # Update portfolio state for this timestamp even without transactions
                    self.portfolio.update_from_transactions(timestamp, pl.DataFrame(), timestamp_data)
                    
                    # Call on_market_close at the end of the trading day
                    # Assuming last timestamp of the day is market close
                    is_last_timestamp = timestamp == daily_data.select(pl.col('timestamp').max()).item()
                    if is_last_timestamp and hasattr(self.strategy, 'on_market_close'):
                        self.strategy.on_market_close(context, timestamp_data, self.portfolio)
            
            # Prepare results
            return {
                'stats': self.portfolio.get_stats(),
                'equity_curve': self.portfolio.get_equity_curve(),
                'transactions': self.portfolio.get_transactions(),
                'positions': self.portfolio.get_positions(),
                'success': True
            }
            
        finally:
            self._create_empty_results()
    
    def _create_transactions_from_signals(self, signals, market_data):
        """Convert signals to transactions.
        
        Args:
            signals: List of Signal objects
            market_data: DataFrame with current market data
            
        Returns:
            DataFrame with transactions
        """
        if not signals:
            return pl.DataFrame()
            
        transactions = []
        
        # Create a lookup for current prices
        price_lookup = {}
        if not market_data.is_empty() and 'symbol' in market_data.columns and 'close' in market_data.columns:
            price_lookup = {row['symbol']: row['close'] for row in market_data.select(['symbol', 'close']).to_dicts()}
        
        for signal in signals:
            try:
                # Check if signal is valid
                if not hasattr(signal, 'symbol') or not hasattr(signal, 'action') or not hasattr(signal, 'quantity'):
                    self._logger.warning(f"Invalid signal: {signal}")
                    continue
                
                symbol = signal.symbol
                action = signal.action
                quantity = signal.quantity
                
                # Skip if symbol not in current market data
                if symbol not in price_lookup:
                    self._logger.warning(f"Symbol {symbol} not in current market data")
                    continue
                    
                # Get execution price
                price = signal.price if hasattr(signal, 'price') and signal.price is not None else price_lookup[symbol]
                
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
            except Exception as e:
                self._logger.error(f"Error processing signal: {e}")
        
        return pl.DataFrame(transactions) if transactions else pl.DataFrame()
    
    def _create_empty_results(self):
        """Create empty results dictionary for when no data is available."""
        return {
            'stats': {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0
            },
            'equity_curve': pl.DataFrame(schema={'timestamp': pl.Datetime, 'equity': pl.Float64, 'cash': pl.Float64}),
            'transactions': pl.DataFrame(),
            'positions': pl.DataFrame(),
            'execution_time': 0.0,
            'success': False
        } 
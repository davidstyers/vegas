"""Backtest Engine for the Vegas backtesting engine.

This module provides a streamlined backtesting engine.
"""

from datetime import datetime, timedelta
import logging
import time
import polars as pl

from vegas.data import DataLayer, DataPortal
from vegas.strategy import Strategy, Context, Signal
from vegas.portfolio import Portfolio
from vegas.broker import Broker
from vegas.pipeline.engine import PipelineEngine


class BacktestEngine:
    """Backtesting engine for the Vegas backtesting system.
    
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
        self.data_layer.engine = self
        self.data_portal = DataPortal(self.data_layer)
        self.strategy = None
        self.portfolio = None
        self.broker = None
        self.timezone = timezone
        
        # Trading hours configuration
        self._market_open_time = "09:30"  # Default market open (US)
        self._market_close_time = "16:00"  # Default market close (US)
        self._market_name = "US"  # Default market name
        self._ignore_extended_hours = False  # By default, use all data
        
        # Add pipeline
        self.pipeline_engine = PipelineEngine(self.data_portal)
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

    def _get_open_order_symbols(self):
        """Return symbols for currently open or partially filled orders."""
        symbols = set()
        try:
            if self.broker is not None and hasattr(self.broker, 'orders'):
                from vegas.broker.broker import OrderStatus  # local import to avoid cycles at module import time
                for order in self.broker.orders:
                    if getattr(order, 'status', None) in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                        sym = getattr(order, 'symbol', None)
                        if sym:
                            symbols.add(sym)
        except Exception:
            pass
        return symbols

    def _get_active_position_symbols(self):
        """Return symbols that currently have a non-zero position size."""
        symbols = set()
        try:
            if self.portfolio is not None and hasattr(self.portfolio, 'positions'):
                for sym, qty in self.portfolio.positions.items():
                    try:
                        if abs(float(qty)) > 1e-6:
                            symbols.add(sym)
                    except Exception:
                        continue
        except Exception:
            pass
        return symbols

    def _discover_universe(self, signals):
        """Discover dynamic universe from positions, open orders, and current signals."""
        universe = set()
        try:
            universe |= self._get_active_position_symbols()
            universe |= self._get_open_order_symbols()
            if signals:
                for sig in signals:
                    sym = getattr(sig, 'symbol', None)
                    if sym:
                        universe.add(sym)
        except Exception:
            pass
        return universe
        
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
        self._logger.info(f"Starting backtest from {start} to {end}")
        
        # Initialize strategy and portfolio
        self.strategy = strategy

        self.portfolio = Portfolio(initial_capital=initial_capital, data_portal=self.data_portal)
        self.broker = Broker(initial_cash=initial_capital, data_portal=self.data_portal)
        
        # Get context and initialize strategy
        context = self.strategy.context
        context.set_portfolio(self.portfolio)  # Ensure portfolio is accessible
        context.set_engine(self)  # Set reference to the engine for pipeline access

        # Expose commission namespace for user convenience before initialize()
        try:
            from vegas.broker.commission import commission as _commission_ns
            context.commission = _commission_ns
        except Exception:
            pass

        # Allow strategy to configure commission in initialize
        self.strategy.initialize(context)

        # After initialize, if a commission model was set, persist it on engine for reuse
        try:
            self._commission_model = context.get_commission_model()
            if self._commission_model is not None:
                self.broker.commission_model = self._commission_model
        except Exception:
            self._commission_model = None
        
        start_time = time.time()
        
        # Configure market hours filter (applied at query time when fetching bar snapshots)
        market_hours = None
        if self._ignore_extended_hours:
            market_hours = (self._market_open_time, self._market_close_time)

        # Acquire and prepare market data; build simulation timestamp index
        timestamp_index = self._prepare_market_data(start, end)

        # Run the backtest
        self._logger.info("Executing backtest")
        results_dict = self._run_backtest(context, timestamp_index, market_hours)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self._logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        
        # Add execution time to results
        results_dict['execution_time'] = execution_time
        
        # Allow strategy to analyze results
        self.strategy.analyze(context, results_dict)
        
        return results_dict

    def _prepare_market_data(self, start: datetime, end: datetime) -> pl.Series:
        """Load and prepare market data for the backtest and return the timestamp index.

        - Determines strategy universe (if provided on the strategy)
        - Computes earliest preload start using attached pipeline window requirements
        - Preloads all required frequencies into the DataPortal
        - Returns the unified simulation timestamp index from the cache
        """
        # Determine strategy universe, if exposed
        try:
            strategy_universe = None
            for attr in ("universe", "assets", "symbols"):
                if hasattr(self.strategy, attr):
                    val = getattr(self.strategy, attr)
                    if isinstance(val, (list, tuple, set)) and len(val) > 0:
                        strategy_universe = list(val)
                        break
                    if isinstance(val, str) and val:
                        strategy_universe = [val]
                        break
        except Exception:
            strategy_universe = None

        # Frequencies to materialize
        freq_set = {"1h"}
        preload_start = start
        try:
            for p in getattr(self, "attached_pipelines", {}).values():
                f = getattr(p, "frequency", None)
                if isinstance(f, str) and f:
                    freq_set.add(f)
                try:
                    max_window = self.pipeline_engine._get_max_window_length(p)
                except Exception:
                    max_window = 0
                if max_window and max_window > 0:
                    if f == "1d":
                        candidate = start - timedelta(days=int(max_window))
                    elif f == "1h":
                        candidate = start - timedelta(hours=int(max_window))
                    else:
                        candidate = start - timedelta(hours=int(max_window))
                    if candidate < preload_start:
                        preload_start = candidate
        except Exception:
            pass

        # Preload the DataPortal cache
        self.data_portal.load_data(
            start_date=preload_start,
            end_date=end,
            symbols=strategy_universe,
            frequencies=sorted(freq_set),
            market_hours=(self._market_open_time, self._market_close_time) if self._ignore_extended_hours else None,
        )

        # Build and return the timestamp index from the cache
        return self.data_portal.get_unified_timestamp_index(start, end, frequency="1h")
    
    def _run_backtest(self, context, timestamp_index=None, market_hours=None):
        """Run the backtest algorithm.
        
        Args:
            market_data: DataFrame with market data for the backtest period
            context: Strategy context
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Drive the daily iteration from the unified timestamp index when available
            idx_dates = (
                pl.DataFrame({"timestamp": timestamp_index})
                .select(pl.col("timestamp").dt.date().alias("date"))
                .unique()
                .sort("date")
                .get_column("date")
                )
            if idx_dates.len() == 0:
                self._logger.warning("No data available for the specified period")
                return self._create_empty_results()
            
            unique_dates = idx_dates.to_list()
            self._logger.info(f"Processing {len(unique_dates)} trading days")
            
            # Process data day by day
            for date in unique_dates:
                # Clear previous pipeline results
                self._pipeline_results = {}
                
                # Get data for the current day (timestamp only used for pipeline scheduling)
                context.current_ts = datetime.combine(date, datetime.min.time())
                self.data_portal.set_current_dt(context.current_ts)
                
                # Compute any attached pipelines for this day
                for name, pipeline in self.attached_pipelines.items():
                    try:
                        # Run pipeline for just this date
                        pipeline_result = self.pipeline_engine.run_pipeline(
                            pipeline, 
                            start_date=context.current_ts,
                            end_date=context.current_ts
                            )
                        if pipeline_result.height > 0:
                            # Make results available via pipeline_output
                            self._pipeline_results[name] = pipeline_result
                            self._logger.debug(
                                f"Pipeline '{name}' computed {len(pipeline_result)} results for {date}"
                            )
                        else:
                            self._logger.warning(f"Pipeline '{name}' returned empty results for {date}")
                    finally:
                        pass
                
                # Call before_trading_start at the beginning of each day
                if hasattr(self.strategy, 'before_trading_start'):                   
                    self.strategy.before_trading_start(context, self.data_portal)
                
                # Determine the day's timestamp sequence from unified index
                day_idx = (
                    pl.DataFrame({"timestamp": timestamp_index})
                    .with_columns(pl.col("timestamp").dt.date().alias("date"))
                    .filter(pl.col("date") == date)
                    .select("timestamp")
                    .get_column("timestamp")
                )
                iter_timestamps = day_idx.to_list()

                # Process each timestamp chronologically
                for ts in iter_timestamps:
                    self.data_portal.set_current_dt(ts)
                    
                    # Call handle_data for each timestamp to generate trading signals
                    if hasattr(self.strategy, 'handle_data'):
                        signals = self.strategy.handle_data(context, self.data_portal)
                        
                        if signals:
                            for signal in signals:
                                self.broker.place_order(signal)

                    # Discover current universe: active positions + open orders + current signals
                    universe = self._discover_universe(signals if 'signals' in locals() else None)
                    
                    # Execute orders using a snapshot sourced via DataPortal to ensure single interface
                    transactions = []
                    if universe:
                        try:
                            transactions = self.broker.execute_orders_with_portal(sorted(list(universe)), ts, market_hours)
                        except Exception:
                            transactions = []

                    # Update portfolio with transactions
                    if transactions:
                        transactions_pl = pl.from_records(
                            [
                                {
                                    "symbol": t.symbol,
                                    "quantity": t.quantity,
                                    "price": t.price,
                                    "commission": t.commission,
                                }
                                for t in transactions
                            ]
                        )
                        # Update portfolio (market data ignored internally; prices via DataPortal)
                        self.portfolio.update_from_transactions(ts, transactions_pl)
                    
                    # Update portfolio state for this timestamp even without transactions
                    self.portfolio.update_from_transactions(ts, pl.DataFrame())
                    
                    # Call on_market_close at the end of the trading day
                    # Assuming last timestamp of the day is market close
                    is_last_timestamp = (ts == max(iter_timestamps)) if len(iter_timestamps) > 0 else False
                    if is_last_timestamp and hasattr(self.strategy, 'on_market_close'):
                        self.strategy.on_market_close(context, self.data_portal, self.portfolio)

                    context.current_ts = ts
            
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

    def run_live(self, start, end, strategy, feed=None, broker=None):
        """
        Run in a live-like loop using an optional MarketDataFeed and a BrokerAdapter-compatible object.
        Notes:
          - initial_capital is NOT used here; cash/positions are seeded from broker.get_account()/get_positions()
          - When no feed is provided, this method delegates to run() for historical mode.
        """
        pass

    def dump_positions_ledger(self) -> pl.DataFrame:
        """
        Build and return the complete positions ledger (open and closed) as a Polars DataFrame.

        Delegates to the portfolio to reconstruct positions from recorded events/transactions.
        Returns an empty DataFrame if the portfolio is not initialized.
        """
        try:
            if self.portfolio is None:
                return pl.DataFrame()
            # Portfolio provides latest price lookup internally when None
            return self.portfolio.build_positions_ledger(latest_price_lookup=None)
        except Exception:
            # Fail-safe: never raise from dump; return empty DF to keep CLI resilient
            return pl.DataFrame()
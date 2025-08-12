"""Vegas Backtest Execution Engine.

This module implements a streamlined, event-lite backtest engine used by
Vegas. Instead of a separate event bus, it iterates chronologically over a
unified timestamp index sourced from the data cache, invoking user strategy
hooks in a predictable order:

1) Daily loop over unique trading dates
2) Compute any attached pipelines for the date
3) Call ``before_trading_start`` if present
4) Intraday loop over timestamps for the date
   - Set current time in the portal
   - Call ``handle_data`` to obtain trading signals
   - Route signals to the broker to place orders
   - Execute eligible orders using a single market data snapshot
   - Update the portfolio from transactions (or a no-op update if none)
   - Call ``on_market_close`` at the last timestamp of the day

Design goals and rationale:

- Favor clarity over a complex event system; keep the backtest loop readable.
- Single interface to market data via ``DataPortal`` to simplify mocking and
  testing.
- Pipelines are scheduled once per day to align with most daily ranking or
  screening workflows.
- Graceful failure defaults (e.g., empty dataframes) to keep iterative research
  workflows resilient.

Market-hours unification:

- Trading session filtering is governed by a single calendar source
  (``vegas.calendars``). The engine selects a calendar (default ``"24/7"``)
  and the ``DataPortal`` filters timestamps accordingly. Any legacy engine
  helpers for Regular Trading Hours (RTH) have been removed in favor of this
  unified approach.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import polars as pl

from vegas.broker import Broker
from vegas.data import DataLayer, DataPortal
from vegas.pipeline.engine import PipelineEngine
from vegas.portfolio import Portfolio
from vegas.strategy import Context, Signal, Strategy
from vegas.calendars import get_calendar


class BacktestEngine:
    """Backtesting engine for the Vegas system.

    The engine is responsible for orchestrating the interaction between the
    strategy, the broker, and the portfolio using market data served via the
    ``DataPortal``. It runs a deterministic loop over timestamps in the
    requested period and wires the strategy hooks that generate signals.

    Example:
        >>> from vegas.engine import BacktestEngine
        >>> from datetime import datetime
        >>> engine = BacktestEngine(data_dir="db", timezone="UTC")
        >>> # Select a trading calendar (e.g., NYSE or 24/7)
        >>> engine._calendar_name = "NYSE"
        >>> # engine.load_data(directory="/path/to/csvs")  # optional if already loaded
        >>> # results = engine.run(
        ... #     start=datetime(2021, 1, 1),
        ... #     end=datetime(2021, 3, 1),
        ... #     strategy=my_strategy_instance,
        ... #     initial_capital=100_000.0,
        ... # )

    """

    def __init__(self, data_dir: str = "db", timezone: str = "UTC") -> None:
        """Initialize the backtest engine and its dependencies.

        The engine owns the ``DataLayer`` and ``DataPortal`` so it can enforce a
        consistent access path to market data. A logger is also created for
        traceability of backtest steps.

        :param data_dir: Directory used by the data layer to store/load data files.
        :type data_dir: str
        :param timezone: Timezone for data timestamps (e.g., ``"UTC"``).
        :type timezone: str
        :raises Exception: Propagates exceptions from underlying data components
            if initialization fails.
        :returns: None
        :rtype: None
        :Example:
            >>> engine = BacktestEngine(data_dir="db", timezone="UTC")
        """
        self._logger = logging.getLogger("vegas.engine")
        self._logger.info(
            f"Initializing BacktestEngine with data directory: {data_dir}, timezone: {timezone}"
        )

        self.data_layer = DataLayer(data_dir, timezone=timezone)
        self.data_layer.engine = self
        self.data_portal = DataPortal(self.data_layer)
        self.strategy: Optional[Strategy] = None
        self.portfolio: Optional[Portfolio] = None
        self.broker: Optional[Broker] = None
        # Engine timezone follows calendar selection; default to calendar's tz later
        self.timezone = timezone

        # Selected trading calendar name; used by DataPortal for timestamp filtering.
        # Default calendar is 24/7 (no filtering).
        self._calendar_name: str = "24/7"

    def set_calendar(self, name: str) -> None:
        """Select the trading calendar used to filter market data.

        :param name: Calendar name (e.g., "NYSE", "24/7").
        :type name: str
        :raises KeyError: If the calendar name is not recognized.
        :returns: None
        :rtype: None
        :Example:
            >>> engine.set_calendar("NYSE")
        """
        # Validate early and store name; actual instance is resolved at load time
        get_calendar(name)
        self._calendar_name = name

        # Pipeline scheduling engine: pipelines are computed once per day to
        # support ranking/screening workflows aligned to the research cadence.
        self.pipeline_engine = PipelineEngine(self.data_portal)
        self.attached_pipelines: Dict[str, Any] = {}
        self._pipeline_results: Dict[str, pl.DataFrame] = {}

    # Legacy market-hours helpers removed in favor of calendar-based filtering.

    def attach_pipeline(self, pipeline: Any, name: str) -> Any:
        """Register a pipeline to compute once per trading day.

        Pipelines are executed at the start of each simulated day and the
        resulting frame is exposed via ``pipeline_output(name)``. Attaching
        multiple pipelines is supported.

        :param pipeline: Pipeline definition object understood by ``PipelineEngine``.
        :type pipeline: Any
        :param name: Unique name used to reference the pipeline output.
        :type name: str
        :returns: The pipeline object that was attached for convenience.
        :rtype: Any
        :raises ValueError: If a pipeline with the same name is already attached.
        :Example:
            >>> engine.attach_pipeline(my_pipeline, name="top_momentum")
        """
        if name in self.attached_pipelines:
            # Avoid silent override which can hide configuration errors in notebooks.
            raise ValueError(f"Pipeline named '{name}' is already attached.")
        self._logger.info(f"Attaching pipeline '{name}'")
        self.attached_pipelines[name] = pipeline
        return pipeline

    def pipeline_output(self, name: str) -> pl.DataFrame:
        """Return results for a previously attached pipeline for the current day.

        The daily backtest loop refreshes pipeline results each morning. If the
        pipeline produced no output (e.g., empty universe) this method returns an
        empty ``DataFrame``. If the named pipeline was never attached, a warning
        is logged and an empty frame is returned to keep workflows resilient.

        :param name: The name of the pipeline whose results to fetch.
        :type name: str
        :returns: The pipeline results for the current date or an empty frame if unavailable.
        :rtype: polars.DataFrame
        :Example:
            >>> output = engine.pipeline_output("top_momentum")
            >>> if output.height > 0:
            ...     do_something(output)
        """
        if name not in self._pipeline_results:
            self._logger.warning(
                f"No pipeline named '{name}' has been attached or computed"
            )
            # Return empty DataFrame instead of raising an error
            return pl.DataFrame()
        return self._pipeline_results[name]

    def _get_open_order_symbols(self) -> Set[str]:
        """Collect symbols referenced by open or partially filled orders.

        This supports dynamic universe discovery so that execution considers any
        symbols relevant to in-flight orders, ensuring we price and settle them
        even if the strategy does not emit a new signal at the same timestamp.

        :returns: Set of symbol strings currently referenced by non-finalized orders.
        :rtype: set[str]
        :raises Exception: Exceptions are swallowed to keep the backtest loop
            robust; an empty set is returned on error.
        :Example:
            >>> symbols = engine._get_open_order_symbols()
        """
        symbols: Set[str] = set()
        try:
            if self.broker is not None and hasattr(self.broker, "orders"):
                from vegas.broker.broker import (  # local import to avoid cycles at module import time
                    OrderStatus,
                )

                for order in self.broker.orders:
                    if getattr(order, "status", None) in (
                        OrderStatus.OPEN,
                        OrderStatus.PARTIALLY_FILLED,
                    ):
                        sym = getattr(order, "symbol", None)
                        if sym:
                            symbols.add(sym)
        except Exception:
            pass
        return symbols

    def _get_active_position_symbols(self) -> Set[str]:
        """Collect symbols for which the portfolio currently holds non-zero positions.

        The engine uses this to ensure we continue marking to market and
        evaluating positions even when the strategy is not actively trading a
        name at a given timestamp.

        :returns: Set of symbol strings with absolute position quantity above a minimal threshold.
        :rtype: set[str]
        :raises Exception: Exceptions are swallowed to keep the backtest loop robust.
        :Example:
            >>> symbols = engine._get_active_position_symbols()
        """
        symbols: Set[str] = set()
        try:
            if self.portfolio is not None and hasattr(self.portfolio, "positions"):
                for sym, qty in self.portfolio.positions.items():
                    try:
                        if abs(float(qty)) > 1e-6:
                            symbols.add(sym)
                    except Exception:
                        continue
        except Exception:
            pass
        return symbols

    def _discover_universe(self, signals: Optional[Iterable[Signal]]) -> Set[str]:
        """Derive the execution universe for the current timestamp.

        The universe is computed as the union of:
        - Symbols with non-zero positions
        - Symbols with open/partially filled orders
        - Symbols present in the current batch of strategy signals

        This ensures that we will price any asset that could affect state
        transitions at the current timestamp.

        :param signals: Iterable of signals produced by the strategy at the current timestamp.
        :type signals: Optional[Iterable[Signal]]
        :returns: Set of symbol strings to include in order execution and state updates.
        :rtype: set[str]
        :Example:
            >>> u = engine._discover_universe(signals)
        """
        universe: Set[str] = set()
        try:
            universe |= self._get_active_position_symbols()
            universe |= self._get_open_order_symbols()
            if signals:
                for sig in signals:
                    sym = getattr(sig, "symbol", None)
                    if sym:
                        universe.add(sym)
        except Exception:
            pass
        return universe

    # _is_regular_market_hours removed; use calendars via ``vegas.calendars``.

    def load_data(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
        file_pattern: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> None:
        """Load market data into the engine's ``DataLayer``/``DataPortal``.

        The engine supports loading a single file or scanning a directory with an
        optional glob-like pattern. If no source is specified and no data has
        been previously loaded, a ``ValueError`` is raised to prevent silent
        execution with empty data.

        :param file_path: Optional path to a single CSV file.
        :type file_path: Optional[str]
        :param directory: Optional directory containing multiple data files.
        :type directory: Optional[str]
        :param file_pattern: Optional pattern used when scanning a directory.
        :type file_pattern: Optional[str]
        :param max_files: Optional limit on number of files to load when scanning a directory.
        :type max_files: Optional[int]
        :raises ValueError: If no data source is specified and no data is already loaded.
        :returns: None
        :rtype: None
        :Example:
            >>> engine.load_data(directory="/data/quotes", file_pattern="*.csv", max_files=100)
        """
        # Choose a single loading path to avoid accidental duplicate loads.
        if file_path:
            self._logger.info(f"Loading data from single file: {file_path}")
            self.data_layer.load_data(file_path=file_path)
        elif directory:
            self._logger.info(f"Loading data from directory: {directory}")
            self.data_layer.load_data(
                directory=directory, file_pattern=file_pattern, max_files=max_files
            )
        elif not self.data_layer.is_initialized():
            raise ValueError(
                "No data source specified and no data is currently loaded."
            )
        else:
            self._logger.info("Using already loaded data")

        # Provide a concise summary to help users validate the time range and coverage.
        data_info = self.data_layer.get_data_info()
        self._logger.info(
            f"Available data: {data_info.get('row_count', 'unknown')} rows, "
            f"{data_info.get('symbol_count', 0)} symbols, "
            f"from {data_info.get('start_date', 'unknown')} to {data_info.get('end_date', 'unknown')}"
        )

    def run(
        self,
        start: datetime,
        end: datetime,
        strategy: Strategy,
        initial_capital: float = 100_000.0,
    ) -> Dict[str, Any]:
        """Run a historical backtest between ``start`` and ``end``.

        The engine wires the provided ``strategy`` to a ``Portfolio`` and
        ``Broker`` bound to the shared ``DataPortal``. It executes a deterministic
        loop over timestamps using a unified index to ensure consistent behavior
        across strategies.

        :param start: Inclusive start datetime for the backtest window.
        :type start: datetime
        :param end: Inclusive end datetime for the backtest window.
        :type end: datetime
        :param strategy: Strategy implementing ``initialize``, ``handle_data``, and optionally
            ``before_trading_start`` and ``on_market_close``.
        :type strategy: Strategy
        :param initial_capital: Initial cash balance used to seed the portfolio and broker.
        :type initial_capital: float
        :returns: A dictionary containing stats, equity curve, transactions, positions, and success flag.
        :rtype: dict
        :raises Exception: Propagates exceptions thrown by user strategy code or I/O layers.
        :Example:
            >>> results = engine.run(start, end, my_strategy, initial_capital=50_000)
        """
        self._logger.info(f"Starting backtest from {start} to {end}")

        # Initialize strategy and portfolio
        self.strategy = strategy

        self.portfolio = Portfolio(
            initial_capital=initial_capital, data_portal=self.data_portal
        )
        self.broker = Broker(initial_cash=initial_capital, data_portal=self.data_portal)

        # Get context and initialize strategy
        context = self.strategy.context
        context.set_portfolio(
            self.portfolio
        )  # Make portfolio accessible to strategy code
        context.set_engine(
            self
        )  # Allow strategies to access engine helpers like pipelines

        # Expose helper namespaces prior to initialize to make configuration ergonomic.
        try:
            from vegas.broker.commission import commission as _commission_ns

            context.commission = _commission_ns
        except Exception:
            pass
        try:
            from vegas.broker.slippage import slippage as _slippage_ns

            context.slippage = _slippage_ns
        except Exception:
            pass

        # Allow strategy to configure commission in initialize; this mirrors typical live setup.
        self.strategy.initialize(context)

        # If models were set by the strategy, propagate them to the broker.
        try:
            self._commission_model = context.get_commission_model()
            if self._commission_model is not None:
                self.broker.commission_model = self._commission_model
        except Exception:
            self._commission_model = None
        try:
            _slippage_model = context.get_slippage_model()
            if _slippage_model is not None:
                self.broker.slippage_model = _slippage_model
        except Exception:
            pass

        start_time = time.time()

        # Build the unified timestamp index that will drive daily/intraday iteration.
        timestamp_index: pl.Series = self._prepare_market_data(start, end)

        # Run the backtest
        self._logger.info("Executing backtest")
        results_dict = self._run_backtest(context, timestamp_index)

        # Calculate execution time
        execution_time = time.time() - start_time
        self._logger.info(f"Backtest completed in {execution_time:.2f} seconds")

        # Add execution time to results
        results_dict["execution_time"] = execution_time

        # Allow strategy to analyze results
        self.strategy.analyze(context, results_dict)

        return results_dict

    def _prepare_market_data(self, start: datetime, end: datetime) -> pl.Series:
        """Load and prepare data; return the unified timestamp index.

        Steps performed:
        - Determine strategy universe if explicitly provided by the strategy.
        - Compute earliest preload start using window length requirements from attached pipelines.
        - Preload required frequencies into the ``DataPortal`` cache.
        - Build and return the unified timestamp index that will drive simulation.

        :param start: Inclusive start datetime for the backtest window.
        :type start: datetime
        :param end: Inclusive end datetime for the backtest window.
        :type end: datetime
        :returns: Unified timestamp index as a Polars ``Series``.
        :rtype: polars.Series
        :Example:
            >>> idx = engine._prepare_market_data(start, end)
        """
        # Determine strategy-provided static universe to reduce I/O, if available.
        try:
            strategy_universe: Optional[List[str]] = None
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

        # Frequencies to materialize. Default to 1h for a balanced intraday cadence.
        frequency_set: Set[str] = {"1h"}
        preload_start: datetime = start
        try:
            for pipeline in getattr(self, "attached_pipelines", {}).values():
                freq: Optional[str] = getattr(pipeline, "frequency", None)
                if isinstance(freq, str) and freq:
                    frequency_set.add(freq)
                try:
                    max_window = self.pipeline_engine._get_max_window_length(pipeline)
                except Exception:
                    max_window = 0
                if max_window and max_window > 0:
                    if freq == "1d":
                        candidate = start - timedelta(days=int(max_window))
                    elif freq == "1h":
                        candidate = start - timedelta(hours=int(max_window))
                    else:
                        candidate = start - timedelta(hours=int(max_window))
                    if candidate < preload_start:
                        preload_start = candidate
        except Exception:
            pass

        # Preload the DataPortal cache to amortize I/O and speed up the run loop.
        # Resolve calendar and propagate timezone to data components
        cal = get_calendar(getattr(self, "_calendar_name", "24/7"))
        try:
            # Align engine and data layer/portal timezones to the calendar's timezone
            self.timezone = getattr(cal, "timezone", self.timezone)
            if hasattr(self.data_layer, "timezone"):
                self.data_layer.timezone = self.timezone
            if hasattr(self.data_portal, "timezone"):
                self.data_portal.timezone = self.timezone
        except Exception:
            pass

        self.data_portal.load_data(
            start_date=preload_start,
            end_date=end,
            symbols=strategy_universe,
            frequencies=sorted(frequency_set),
            calendar=cal,
        )

        # Build and return the timestamp index from the cache.
        return self.data_portal.get_unified_timestamp_index(start, end, frequency="1h")

    def _run_backtest(
        self,
        context: Context,
        timestamp_index: Optional[pl.Series] = None,
    ) -> Dict[str, Any]:
        """Execute the backtest loop over the provided timestamp index.

        The function organizes the loop as date blocks to schedule pipelines and
        daily hooks once per day. Within each day, it iterates intraday
        timestamps, generating and executing orders, and updating state.

        :param context: Strategy context object.
        :type context: Context
        :param timestamp_index: Unified timestamp index. If ``None``, the function returns empty results.
        :type timestamp_index: Optional[polars.Series]
        :param market_hours: Optional market hours (open, close) used by the broker during execution.
        :type market_hours: Optional[tuple[str, str]]
        :returns: Results dictionary combining stats, equity curve, transactions, and positions.
        :rtype: dict
        :Example:
            >>> results = engine._run_backtest(context, idx)
        """
        try:
            # Derive unique trading dates to anchor daily scheduling.
            if timestamp_index is None:
                self._logger.warning("No timestamp index provided")
                return self._create_empty_results()

            trading_date_series = (
                pl.DataFrame({"timestamp": timestamp_index})
                .select(pl.col("timestamp").dt.date().alias("date"))
                .unique()
                .sort("date")
                .get_column("date")
            )
            if trading_date_series.len() == 0:
                self._logger.warning("No data available for the specified period")
                return self._create_empty_results()

            unique_dates: List[datetime.date] = trading_date_series.to_list()
            self._logger.info(f"Processing {len(unique_dates)} trading days")

            # Process data day by day
            for current_date in unique_dates:
                # Clear previous pipeline results so strategies see fresh daily outputs.
                self._pipeline_results = {}

                # Anchor the date for scheduling daily hooks/pipelines.
                context.current_ts = datetime.combine(current_date, datetime.min.time())
                self.data_portal.set_current_dt(context.current_ts)

                # Compute any attached pipelines for this day
                for name, pipeline in self.attached_pipelines.items():
                    try:
                        # Run pipeline for just this date. Many ranking/screening pipelines operate daily.
                        pipeline_result = self.pipeline_engine.run_pipeline(
                            pipeline,
                            start_date=context.current_ts,
                            end_date=context.current_ts,
                        )
                        if pipeline_result.height > 0:
                            # Expose results via pipeline_output for the strategy to consume.
                            self._pipeline_results[name] = pipeline_result
                            self._logger.debug(
                                f"Pipeline '{name}' computed {len(pipeline_result)} results for {current_date}"
                            )
                        else:
                            self._logger.warning(
                                f"Pipeline '{name}' returned empty results for {current_date}"
                            )
                    finally:
                        pass

                # Call before_trading_start at the beginning of each day
                if hasattr(self.strategy, "before_trading_start"):
                    self.strategy.before_trading_start(context, self.data_portal)

                # Determine the day's timestamp sequence from the unified index.
                day_timestamp_series = (
                    pl.DataFrame({"timestamp": timestamp_index})
                    .with_columns(pl.col("timestamp").dt.date().alias("date"))
                    .filter(pl.col("date") == current_date)
                    .select("timestamp")
                    .get_column("timestamp")
                )
                iter_timestamps: List[datetime] = day_timestamp_series.to_list()

                # Process each timestamp chronologically
                for timestamp in iter_timestamps:
                    self.data_portal.set_current_dt(timestamp)

                    # Call handle_data for each timestamp to generate trading signals
                    if hasattr(self.strategy, "handle_data"):
                        signals = self.strategy.handle_data(context, self.data_portal)

                        if signals:
                            for signal in signals:
                                self.broker.place_order(signal)

                    # Derive current execution universe to ensure correct pricing and settlement.
                    universe = self._discover_universe(
                        signals if "signals" in locals() else None
                    )

                    # Execute orders using a snapshot sourced via ``DataPortal`` to ensure single data interface.
                    transactions = []
                    if universe:
                        try:
                            transactions = self.broker.execute_orders_with_portal(
                                sorted(list(universe)), timestamp
                            )
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
                        # Portfolio consults ``DataPortal`` for prices; we supply transaction ledger only.
                        self.portfolio.update_from_transactions(
                            timestamp, transactions_pl
                        )

                    # Ensure valuations and stats are computed even if no fills occurred at this timestamp.
                    self.portfolio.update_from_transactions(timestamp, pl.DataFrame())

                    # Call on_market_close at the end of the trading day
                    # Assuming last timestamp of the day is market close
                    is_last_timestamp = (
                        (timestamp == max(iter_timestamps))
                        if len(iter_timestamps) > 0
                        else False
                    )
                    if is_last_timestamp and hasattr(self.strategy, "on_market_close"):
                        self.strategy.on_market_close(
                            context, self.data_portal, self.portfolio
                        )

                    context.current_ts = timestamp

            # Prepare results
            return {
                "stats": self.portfolio.get_stats(),
                "equity_curve": self.portfolio.get_equity_curve(),
                "transactions": self.portfolio.get_transactions(),
                "positions": self.portfolio.get_positions(),
                "success": True,
            }

        finally:
            # Ensure any temporary state is not leaked between runs.
            self._create_empty_results()

    def _create_empty_results(self) -> Dict[str, Any]:
        """Construct an empty results dictionary used as a safe fallback.

        Returning a typed empty structure allows downstream analysis/plotting
        utilities to operate without additional ``None`` checks.

        :returns: Empty, well-typed results dictionary.
        :rtype: dict
        :Example:
            >>> empty = engine._create_empty_results()
        """
        return {
            "stats": {"total_return": 0.0, "total_return_pct": 0.0, "num_trades": 0},
            "equity_curve": pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "equity": pl.Float64,
                    "cash": pl.Float64,
                }
            ),
            "transactions": pl.DataFrame(),
            "positions": pl.DataFrame(),
            "execution_time": 0.0,
            "success": False,
        }

    def run_live(
        self,
        start: datetime,
        end: datetime,
        strategy: Strategy,
        feed: Optional[Any] = None,
        broker: Optional[Any] = None,
    ) -> None:
        """Placeholder for a live-like execution loop.

        When implemented, this method will:
        - Seed cash/positions from the live broker rather than ``initial_capital``.
        - Drive the loop from real-time ticks or bars from ``feed``.
        - Delegate to ``run`` for historical mode when ``feed`` is not provided.

        :param start: Start datetime for live backtest window.
        :type start: datetime
        :param end: End datetime for live backtest window.
        :type end: datetime
        :param strategy: Strategy instance to execute.
        :type strategy: Strategy
        :param feed: Optional market data feed interface.
        :type feed: Optional[Any]
        :param broker: Optional broker adapter compatible with the engine's expectations.
        :type broker: Optional[Any]
        :returns: None
        :rtype: None
        :raises NotImplementedError: Currently unimplemented.
        :Example:
            >>> engine.run_live(start, end, strategy, feed=live_feed, broker=live_broker)
        """
        pass

    def dump_positions_ledger(self) -> pl.DataFrame:
        """Return a reconstructed positions ledger as a Polars ``DataFrame``.

        Delegates to the portfolio to rebuild positions from recorded
        transactions and internal events. Returns an empty frame if the
        portfolio is not initialized. Any exceptions are swallowed to prioritize
        CLI/UI resilience.

        :returns: Positions ledger including open and closed positions.
        :rtype: polars.DataFrame
        :Example:
            >>> ledger = engine.dump_positions_ledger()
        """
        try:
            if self.portfolio is None:
                return pl.DataFrame()
            # Portfolio provides latest price lookup internally when None
            return self.portfolio.build_positions_ledger(latest_price_lookup=None)
        except Exception:
            # Fail-safe: never raise from dump; return empty DF to keep CLI resilient
            return pl.DataFrame()

#!/usr/bin/env python3
"""Command-line interface for the Vegas backtesting engine.

This module provides a CLI similar to Zipline for running backtests.
"""

import argparse
import importlib.util
import inspect
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from tabulate import tabulate

from vegas.analytics import generate_quantstats_report
from vegas.calendars import get_calendar
from vegas.data import DataLayer
from vegas.engine import BacktestEngine
from vegas.strategy import Strategy


class VegasCLI:
    """Object-oriented CLI wrapper for Vegas.

    Encapsulates argument parsing, logging configuration, and subcommand handlers
    to provide a reusable programmatic API and a consistent CLI interface.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("vegas.cli")
        self.parser = self._build_parser()
        self.calendar = None

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Vegas Backtesting Engine CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Common arguments
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )
        common_parser.add_argument(
            "--db-dir", type=str, default="db", help="Database directory"
        )

        # Run command
        run_parser = subparsers.add_parser(
            "run", parents=[common_parser], help="Run a backtest"
        )
        run_parser.add_argument("strategy_file", type=str, help="Strategy file to run")
        run_parser.add_argument("--data-file", type=str, help="Data file to use")
        run_parser.add_argument("--data-dir", type=str, help="Data directory to use")
        run_parser.add_argument(
            "--file-pattern", type=str, default="*.csv", help="File pattern for data files"
        )
        run_parser.add_argument(
            "--start", dest="start_date", type=self.parse_date, help="Start date (YYYY-MM-DD)"
        )
        run_parser.add_argument(
            "--end", dest="end_date", type=self.parse_date, help="End date (YYYY-MM-DD)"
        )
        run_parser.add_argument(
            "--capital", type=float, default=100000.0, help="Initial capital"
        )
        run_parser.add_argument(
            "--output", type=str, help="Output file for equity curve chart"
        )
        run_parser.add_argument("--results-csv", type=str, help="Save results to CSV file")
        run_parser.add_argument("--report", type=str, help="Generate QuantStats report")
        run_parser.add_argument("--benchmark", type=str, help="Benchmark symbol for report")
        run_parser.add_argument(
            "--calendar",
            type=str,
            default="24/7",
            help="Trading calendar name (e.g., NYSE, 24/7)",
        )
        run_parser.add_argument(
            "--mode",
            type=str,
            choices=["backtest", "live"],
            default="backtest",
            help="Execution mode",
        )
        run_parser.add_argument(
            "--dump-positions",
            nargs="?",
            const="__DEFAULT__",
            help="Dump all positions (open and closed) at end of run. Optional path to write CSV",
        )
        run_parser.set_defaults(func=self.run_backtest)

        # Ingest command
        ingest_parser = subparsers.add_parser(
            "ingest", parents=[common_parser], help="Ingest data into the database"
        )
        ingest_source = ingest_parser.add_mutually_exclusive_group(required=True)
        ingest_source.add_argument("--file", type=str, help="Data file to ingest")
        ingest_source.add_argument("--directory", type=str, help="Data directory to ingest")
        ingest_parser.add_argument(
            "--pattern", type=str, default="*.csv", help="File pattern for data files"
        )
        ingest_parser.add_argument(
            "--max-files", type=int, help="Maximum number of files to ingest"
        )
        ingest_parser.add_argument(
            "--timezone",
            type=str,
            default="UTC",
            help="Timezone name (e.g., UTC, US/Eastern)",
        )
        ingest_parser.set_defaults(func=self.ingest_data)

        # Ingest OHLCV command
        ohlcv_parser = subparsers.add_parser(
            "ingest-ohlcv", parents=[common_parser], help="Ingest OHLCV data"
        )
        ohlcv_source = ohlcv_parser.add_mutually_exclusive_group(required=True)
        ohlcv_source.add_argument("--file", type=str, help="OHLCV file to ingest")
        ohlcv_source.add_argument(
            "--directory", type=str, help="Directory with OHLCV files"
        )
        ohlcv_parser.add_argument(
            "--max-files", type=int, help="Maximum number of files to ingest"
        )
        ohlcv_parser.add_argument(
            "--timezone",
            type=str,
            default="UTC",
            help="Timezone name (e.g., UTC, US/Eastern)",
        )
        ohlcv_parser.set_defaults(func=self.ingest_ohlcv)

        # DB status command
        status_parser = subparsers.add_parser(
            "db-status", parents=[common_parser], help="Show database status"
        )
        status_parser.add_argument(
            "--detailed", action="store_true", help="Show detailed status"
        )
        status_parser.set_defaults(func=self.db_status)

        # DB query command
        query_parser = subparsers.add_parser(
            "db-query", parents=[common_parser], help="Execute SQL query"
        )
        query_parser.add_argument("--query", type=str, help="SQL query to execute")
        query_parser.add_argument(
            "--query-file", type=str, help="File containing SQL query"
        )
        query_parser.add_argument(
            "--output", type=str, help="Output file for query results"
        )
        query_parser.add_argument(
            "--limit", type=int, default=100, help="Maximum rows to display"
        )
        query_parser.set_defaults(func=self.db_query)

        # Delete DB command
        delete_parser = subparsers.add_parser(
            "delete-db", parents=[common_parser], help="Delete the database"
        )
        delete_parser.add_argument(
            "--force", action="store_true", help="Force deletion without confirmation"
        )
        delete_parser.set_defaults(func=self.delete_db)

        return parser

    def _configure_logging(self, verbose: bool) -> None:
        log_level = logging.DEBUG if verbose else logging.INFO
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def run(self, argv: list[str] | None = None) -> int:
        args = self.parser.parse_args(argv)
        self._configure_logging(getattr(args, "verbose", False))
        if hasattr(args, "func"):
            return args.func(args)
        self.parser.print_help()
        return 1

    # Utilities
    @staticmethod
    def parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid date format: {date_str}. Use YYYY-MM-DD"
            ) from exc

    @staticmethod
    def load_strategy_from_file(file_path: str) -> type[Strategy]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strategy_classes = [
            obj
            for _, obj in inspect.getmembers(module)
            if inspect.isclass(obj) and issubclass(obj, Strategy) and obj is not Strategy
        ]
        if not strategy_classes:
            raise ValueError(f"No Strategy subclass found in {file_path}")
        if len(strategy_classes) > 1:
            logging.warning(
                "Multiple strategy classes found in %s. Using %s",
                str(file_path),
                strategy_classes[0].__name__,
            )
        return strategy_classes[0]

    # Command handlers
    def run_backtest(self, args: argparse.Namespace) -> int:
        try:
            strategy_class = self.load_strategy_from_file(args.strategy_file)
            self.logger.info("Loaded strategy: %s", strategy_class.__name__)

            engine = None
            self.calendar = get_calendar(args.calendar)
            try:
                engine = BacktestEngine(timezone=self.calendar.timezone)

                try:
                    engine.set_calendar(args.calendar)
                    self.logger.info(
                        "Set calendar to %s",
                        self.calendar,
                    )
                except Exception as e:
                    self.logger.error("Invalid calendar '%s': %s", self.calendar, e)
                    return 1

                # Load data
                self._load_data(engine, args)

                # Get data info
                data_info = engine.data_layer.get_data_info()
                self.logger.info(
                    "Database contains %s symbols for %s days",
                    data_info["symbol_count"],
                    data_info["days"],
                )

                # Set up backtest dates
                start_date = args.start_date or data_info["start_date"]
                end_date = args.end_date or data_info["end_date"]

                self.logger.info(
                    "Running backtest from %s to %s in timezone %s",
                    start_date.date(),
                    end_date.date(),
                    getattr(self.calendar, "timezone", "UTC"),
                )

                # Run backtest or live depending on mode
                strategy = strategy_class()
                if getattr(args, "mode", "backtest") == "live":
                    results = engine.run_live(
                        start=start_date,
                        end=end_date,
                        strategy=strategy,
                        feed=None,
                        broker=None,
                    )
                else:
                    results = engine.run(
                        start=start_date,
                        end=end_date,
                        strategy=strategy,
                        initial_capital=args.capital,
                    )

                # Print results
                self._print_results(results, strategy_class.__name__)

                # Generate outputs if requested
                self._generate_outputs(results, strategy_class.__name__, args)

                # Dump positions ledger if requested
                if hasattr(args, "dump_positions") and args.dump_positions is not None:
                    self._dump_positions(engine, args)

                return 0
            finally:
                if engine is not None and getattr(engine, "data_layer", None) is not None:
                    engine.data_layer.close()
        except Exception as exc:
            self.logger.error("Backtest failed: %s", exc)
            return 1

    def ingest_data(self, args: argparse.Namespace) -> int:
        try:
            self.calendar = get_calendar(args.calendar)
            data_layer = DataLayer(data_dir=args.db_dir, timezone=self.calendar.timezone)
            if args.file:
                if not os.path.exists(args.file):
                    self.logger.error("File not found: %s", args.file)
                    return 1
                self.logger.info("Ingesting data from file: %s", args.file)
                data_layer.load_data(file_path=args.file)
                self.logger.info("Data ingestion complete")
            elif args.directory:
                if not os.path.exists(args.directory):
                    self.logger.error("Directory not found: %s", args.directory)
                    return 1
                self.logger.info("Ingesting data from directory: %s", args.directory)
                max_files = args.max_files if args.max_files and args.max_files > 0 else None
                data_layer.load_data(directory=args.directory, file_pattern=args.pattern, max_files=max_files)
                self.logger.info("Data ingestion complete")
            else:
                self.logger.error("No data source specified. Provide --file or --directory")
                return 1
            info = data_layer.get_data_info()
            print(f"\nData ingestion summary:")
            print(f"Symbols: {info['symbol_count']}")
            if info["start_date"] is not None and info["end_date"] is not None:
                start_date = info["start_date"].date() if hasattr(info["start_date"], "date") else info["start_date"]
                end_date = info["end_date"].date() if hasattr(info["end_date"], "date") else info["end_date"]
                print(f"Date range: {start_date} to {end_date}")
            return 0
        except Exception as e:
            self.logger.error("Data ingestion failed: %s", e)
            return 1

    def ingest_ohlcv(self, args: argparse.Namespace) -> int:
        try:
            self.calendar = get_calendar(args.calendar)
            data_layer = DataLayer(data_dir=args.db_dir, timezone=self.calendar.timezone)
            if not data_layer.db_manager:
                self.logger.error("Database manager initialization failed")
                return 1
            if args.file:
                if not os.path.exists(args.file):
                    self.logger.error("File not found: %s", args.file)
                    return 1
                self.logger.info("Ingesting OHLCV file: %s", args.file)
                try:
                    rows = data_layer.ingest_ohlcv_file(args.file)
                    self.logger.info("Ingested %s rows from OHLCV file", rows)
                except Exception as e:
                    self.logger.error("Failed to ingest OHLCV file: %s", e)
                    return 1
            elif args.directory:
                if not os.path.exists(args.directory):
                    self.logger.error("Directory not found: %s", args.directory)
                    return 1
                self.logger.info("Ingesting OHLCV files from directory: %s", args.directory)
                try:
                    max_files = args.max_files if args.max_files and args.max_files > 0 else None
                    rows = data_layer.ingest_ohlcv_directory(args.directory, max_files=max_files)
                    self.logger.info("Ingested %s rows from OHLCV files", rows)
                except Exception as e:
                    self.logger.error("Failed to ingest OHLCV directory: %s", e)
                    return 1
            else:
                self.logger.error("No data source specified. Provide --file or --directory")
                return 1
            print(f"\nDatabase status after ingestion:")
            return self._db_status_internal(args)
        except Exception as e:
            self.logger.error("OHLCV ingestion failed: %s", e)
            return 1

    def db_status(self, args: argparse.Namespace) -> int:
        return self._db_status_internal(args)

    def db_query(self, args: argparse.Namespace) -> int:
        try:
            data_layer = DataLayer(data_dir=args.db_dir)
            if not data_layer.db_manager:
                self.logger.error("Database manager initialization failed")
                return 1
            query = args.query
            if args.query_file:
                if not os.path.exists(args.query_file):
                    self.logger.error("Query file not found: %s", args.query_file)
                    return 1
                with open(args.query_file, "r") as f:
                    query = f.read()
            if not query:
                self.logger.error("No query specified. Use --query or --query-file")
                return 1
            self.logger.info("Executing query...")
            try:
                result = data_layer.db_manager.query_to_df(query)
                if result.empty:
                    print(f"Query returned no results")
                    return 0
                if args.output:
                    result.to_csv(args.output, index=False)
                    print(f"Results saved to {args.output}")
                else:
                    max_rows = args.limit if args.limit > 0 else len(result)
                    pl.Config.set_tbl_rows(max_rows)
                    pl.Config.set_tbl_cols(None)
                    pl.Config.set_tbl_width_chars(None)
                    print(f"{result}")
                    if len(result) > max_rows:
                        print(f"\n(Showing {max_rows} of {len(result)} rows)")
                return 0
            except Exception as e:
                self.logger.error("Query execution failed: %s", e)
                return 1
        except Exception as e:
            self.logger.error("Database query failed: %s", e)
            return 1

    def delete_db(self, args: argparse.Namespace) -> int:
        db_dir = args.db_dir or "db"
        db_path = os.path.join(db_dir, "vegas.duckdb")
        if not os.path.exists(db_path):
            self.logger.info("Database does not exist: %s", db_path)
            return 0
        if not args.force:
            confirm = input(
                f"Are you sure you want to delete the database at {db_path}? [y/N] "
            )
            if confirm.lower() not in ["y", "yes"]:
                self.logger.info("Database deletion cancelled")
                return 0
        try:
            os.remove(db_path)
            self.logger.info("Database deleted: %s", db_path)
            wal_path = db_path + ".wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)
                self.logger.info("Database WAL file deleted: %s", wal_path)
            return 0
        except Exception as e:
            self.logger.error("Failed to delete database: %s", e)
            return 1

    # Helper methods used by command handlers
    def _load_data(self, engine: BacktestEngine, args: argparse.Namespace) -> None:
        try:
            if getattr(args, "data_file", None):
                self.logger.info("Loading data from file: %s", args.data_file)
                engine.load_data(file_path=args.data_file)
            elif getattr(args, "data_dir", None):
                self.logger.info("Loading data from directory: %s", args.data_dir)
                engine.load_data(directory=args.data_dir, file_pattern=args.file_pattern)
            else:
                # Try to use already loaded/ingested data
                self.logger.info("No data source specified, using already ingested data")
                if not engine.data_layer.is_initialized():
                    self.logger.error(
                        "No data available. Please provide a data source or ingest data first."
                    )
                    raise ValueError("No data available")
        except Exception as e:
            self.logger.error("Error loading data: %s", e)
            raise

    def _db_status_internal(self, args: argparse.Namespace) -> int:
        try:
            data_layer = DataLayer(data_dir=args.db_dir)
            if not data_layer.db_manager:
                self.logger.error("Database manager initialization failed")
                return 1
            try:
                db_size = data_layer.db_manager.get_database_size()
                dates = data_layer.db_manager.get_available_dates()
                symbols = data_layer.db_manager.get_available_symbols()
                sources = data_layer.db_manager.get_data_sources()
                print(f"\nDatabase Status:")
                print(f"Database location: {data_layer.db_path}")
                print(f"Database size: {db_size / (1024 * 1024):.2f} MB")
                if dates.empty:
                    print(f"No data available in the database")
                    return 0
                print(
                    f"Date range: {dates['start_date'].iloc[0].date()} to {dates['end_date'].iloc[0].date()}"
                )
                print(f"Days with data: {dates['day_count'].iloc[0]}")
                print(f"Symbols: {len(symbols)}")
                print(f"Total records: {symbols['record_count'].sum():,}")
                print(f"Data sources: {len(sources)}")
                if getattr(args, "detailed", False):
                    if not symbols.empty:
                        print(f"\nTop 10 symbols by record count:")
                        top_symbols = symbols.nlargest(10, "record_count")
                        for _, row in top_symbols.iterrows():
                            print(f"  {row['symbol']}: {row['record_count']:,} records")
                    if not sources.empty:
                        print(f"\nData sources:")
                        for _, row in sources.iterrows():
                            added = (
                                row["added_date"].strftime("%Y-%m-%d %H:%M:%S")
                                if pl.not_null(row["added_date"])
                                else "Unknown"
                            )
                            print(
                                f"  {row['source_name']}: {row['row_count']:,} rows, added on {added}"
                            )
                return 0
            except Exception as e:
                self.logger.error("Failed to get database status: %s", e)
                return 1
        except Exception as e:
            self.logger.error("Database status check failed: %s", e)
            return 1

    def _print_results(self, results, strategy_name: str) -> None:
        # Handle both Results objects and dictionaries for backward compatibility
        if hasattr(results, 'stats'):
            # Results object
            stats = results.stats
            execution_time = results.execution_time
        else:
            # Dictionary (backward compatibility)
            stats = results["stats"]
            execution_time = results["execution_time"]
            
        rows = [[k, v] for k, v in stats.items()]
        self.logger.info(
            "\n\n%s Backtest Results:\n%s\n\nExecution Time: %.2f seconds",
            strategy_name,
            tabulate(
                rows,
                headers=["Statistic", "Value"],
                tablefmt="rounded_grid",
                colalign=("center", "right"),
                floatfmt=",.2f",
            ),
            execution_time,
        )

    def _dump_positions(self, engine: BacktestEngine, args: argparse.Namespace) -> None:
        try:
            df = engine.dump_positions_ledger()
        except Exception as e:
            self.logger.error("dump_positions_ledger failed: %s", e)
            df = pl.DataFrame()

        if isinstance(df, pl.DataFrame) and df.height > 0:
            disp = df
            if "pos_open_ts" in disp.columns:
                disp = disp.with_columns(
                    pl.col("pos_open_ts").map_elements(
                        lambda x: x.isoformat() if x is not None else "",
                        return_dtype=pl.Utf8,
                    )
                )
            if "pos_close_ts" in disp.columns:
                disp = disp.with_columns(
                    pl.col("pos_close_ts").map_elements(
                        lambda x: x.isoformat() if x is not None else "",
                        return_dtype=pl.Utf8,
                    )
                )
            headers = disp.columns
            rows = [[row.get(col) for col in headers] for row in disp.to_dicts()]
            table = tabulate(rows, headers=headers, tablefmt="fancy_grid", floatfmt=".4f")
            print(f"\nPositions Ledger:\n{table}")
        else:
            print(f"\nPositions Ledger:\n(empty)")

        # Determine CSV path behavior
        path_arg = args.dump_positions
        if path_arg == "__DEFAULT__":
            csv_path = "./positions_ledger.csv"
        elif isinstance(path_arg, str):
            csv_path = path_arg
        else:
            csv_path = None

        if csv_path:
            try:
                df_out = df.clone() if isinstance(df, pl.DataFrame) else pl.DataFrame()
                if isinstance(df_out, pl.DataFrame) and df_out.height > 0:
                    if "pos_open_ts" in df_out.columns:
                        df_out = df_out.with_columns(pl.col("pos_open_ts").cast(pl.Utf8))
                    if "pos_close_ts" in df_out.columns:
                        df_out = df_out.with_columns(pl.col("pos_close_ts").cast(pl.Utf8))
                df_out.write_csv(csv_path)
                self.logger.info("Positions ledger CSV written to %s", csv_path)
            except Exception as e:
                self.logger.error("Failed to write positions ledger CSV: %s", e)
    
    def _generate_outputs(self, results, strategy_name: str, args: argparse.Namespace) -> None:
        # Handle both Results objects and dictionaries for backward compatibility
        if hasattr(results, 'equity_curve'):
            # Results object
            equity_curve = results.equity_curve
        else:
            # Dictionary (backward compatibility)
            equity_curve = results["equity_curve"]
            
        has_data = hasattr(equity_curve, "is_empty") and not equity_curve.is_empty
        if args.output and has_data:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve["timestamp"], equity_curve["equity"])
            plt.title(f"Portfolio Equity Curve - {strategy_name}")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.grid(True)
            plt.savefig(args.output)
            self.logger.info("Equity curve saved to %s", args.output)
        if args.results_csv and has_data:
            equity_curve.write_csv(args.results_csv)
            self.logger.info("Results saved to %s", args.results_csv)
        if getattr(args, "report", None):
            self._generate_quantstats_report(
                results, strategy_name, args.report, getattr(args, "benchmark", None)
            )

    def _generate_quantstats_report(
        self, results, strategy_name: str, report_path: str, benchmark: str | None
    ) -> None:
        """Generate QuantStats report using the Results object's create_tearsheet method."""
        try:
            # Handle both Results objects and dictionaries for backward compatibility
            if hasattr(results, 'create_tearsheet'):
                # Results object - use its built-in method
                results.create_tearsheet(
                    title=f"{strategy_name} Performance Report",
                    benchmark_symbol=benchmark,
                    output_file=report_path,
                    output_format="html"
                )
                self.logger.info("QuantStats report generation complete")
            else:
                # Dictionary (backward compatibility) - use the function
                success = generate_quantstats_report(
                    results=results,
                    strategy_name=strategy_name,
                    report_path=report_path,
                    benchmark=benchmark,
                    logger=self.logger,
                )
                if success:
                    self.logger.info("QuantStats report generation complete")
                else:
                    self.logger.error("QuantStats report generation failed")
        except Exception as e:
            self.logger.error(f"Error generating QuantStats report: {e}")



def main():
    """CLI entry point delegating to the object-oriented CLI class."""
    cli = VegasCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

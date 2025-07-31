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
import pandas as pd
import matplotlib.pyplot as plt

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy
from vegas.data import DataLayer


def load_strategy_from_file(file_path):
    """Load a strategy class from a Python file."""
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {file_path}")
    
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not import {file_path}")
        
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    
    spec.loader.exec_module(module)
    
    # Find all Strategy subclasses in the module
    strategy_classes = [
        obj for name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, Strategy) and obj != Strategy
    ]
    
    if not strategy_classes:
        raise ValueError(f"No Strategy subclass found in {file_path}")
    
    # If multiple strategy classes found, use the first one
    if len(strategy_classes) > 1:
        logging.warning(f"Multiple strategy classes found in {file_path}. Using {strategy_classes[0].__name__}")
    
    return strategy_classes[0]


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def run_backtest(args):
    """Run a backtest with the specified arguments."""
    # Get logger for CLI
    logger = logging.getLogger('vegas.cli')
    
    try:
        # Load strategy from file
        strategy_class = load_strategy_from_file(args.strategy_file)
        logger.info(f"Loaded strategy: {strategy_class.__name__}")
        
        # Initialize engine
        engine = BacktestEngine(timezone=args.timezone)
        
        # Configure trading hours if provided
        if hasattr(args, 'market') or hasattr(args, 'market_open') or hasattr(args, 'market_close'):
            market = getattr(args, 'market', "US")
            market_open = getattr(args, 'market_open', "09:30")
            market_close = getattr(args, 'market_close', "16:00")
            engine.set_trading_hours(market, market_open, market_close)
            logger.info(f"Configured trading hours: {market_open} to {market_close} for {market} market")
        
        # Configure extended hours handling
        if hasattr(args, 'regular_hours_only') and args.regular_hours_only:
            engine.ignore_extended_hours(True)
            logger.info("Extended hours data will be ignored")
        
        # Load data
        load_data(engine, args, logger)
        
        # Get data info
        data_info = engine.data_layer.get_data_info()
        #logger.info(f"Loaded {data_info['row_count']} data points for {data_info['symbol_count']} symbols")
        
        # Set up backtest dates
        start_date = args.start_date or data_info['start_date']
        end_date = args.end_date or data_info['end_date']
        
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()} in timezone {args.timezone}")
        
        # Run backtest
        strategy = strategy_class()
        results = engine.run(
            start=start_date,
            end=end_date,
            strategy=strategy,
            initial_capital=args.capital
        )
        
        # Print results
        print_results(results, strategy_class.__name__)
        
        # Generate outputs if requested
        generate_outputs(results, strategy_class.__name__, args)
        
        return 0
    finally:
        engine.data_layer.close()


def load_data(engine, args, logger):
    """Load data into the engine based on CLI arguments."""
    try:
        if args.data_file:
            logger.info(f"Loading data from file: {args.data_file}")
            engine.load_data(file_path=args.data_file)
        elif args.data_dir:
            logger.info(f"Loading data from directory: {args.data_dir}")
            engine.load_data(directory=args.data_dir, file_pattern=args.file_pattern)
        else:
            # Try to use already loaded/ingested data
            logger.info("No data source specified, using already ingested data")
            if not engine.data_layer.is_initialized():
                logger.error("No data available. Please provide a data source or ingest data first.")
                raise ValueError("No data available")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def print_results(results, strategy_name):
    """Print backtest results to the console."""
    stats = results['stats']
    print("\nBacktest Results:")
    print(f"Strategy: {strategy_name}")
    print(f"Total Return: {stats.get('total_return_pct', 0.0):.2f}%")
    
    # Print additional stats if available
    if 'sharpe_ratio' in stats:
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in stats:
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    
    print(f"Number of Trades: {stats.get('num_trades', 0)}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")


def generate_outputs(results, strategy_name, args):
    """Generate output files based on CLI arguments."""
    logger = logging.getLogger('vegas.cli')
    equity_curve = results['equity_curve']
    
    # Save equity curve plot if output file is specified
    if args.output and not equity_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve['timestamp'], equity_curve['equity'])
        plt.title(f'Portfolio Equity Curve - {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig(args.output)
        logger.info(f"Equity curve saved to {args.output}")
    
    # Save results to CSV if requested
    if args.results_csv and not equity_curve.empty:
        equity_curve.to_csv(args.results_csv, index=False)
        logger.info(f"Results saved to {args.results_csv}")
    
    # Generate QuantStats report if requested
    if args.report:
        generate_quantstats_report(results, strategy_name, args.report, args.benchmark, logger)


def generate_quantstats_report(results, strategy_name, report_path, benchmark, logger):
    """Generate a QuantStats performance report."""
    try:
        import quantstats as qs
        import os
        
        # Prepare returns series for QuantStats
        equity_curve = results['equity_curve']
        returns = equity_curve.set_index('timestamp')['equity']
        
        # Ensure the returns have a proper datetime index
        returns = prepare_returns_for_quantstats(returns, logger)
        
        # Get benchmark data
        benchmark = benchmark or 'SPY'
        logger.info(f"Generating QuantStats report with benchmark {benchmark}...")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
        
        try:
            # Generate the report
            qs.reports.html(returns, benchmark=benchmark, output=report_path, 
                           title=f"{strategy_name} Performance Report")
            
            if os.path.exists(report_path):
                logger.info(f"QuantStats report saved to {report_path}")
            else:
                logger.error(f"Failed to create report at {report_path}")
                
        except Exception as e:
            logger.warning(f"Error with standard report: {e}. Trying a basic report...")
            
            # Try with a simpler report
            try:
                qs.reports.basic(returns, output=report_path, 
                               title=f"{strategy_name} Basic Performance Report")
                
                if os.path.exists(report_path):
                    logger.info(f"Basic QuantStats report saved to {report_path}")
                else:
                    logger.error(f"Failed to create basic report at {report_path}")
            except Exception as e2:
                logger.error(f"Error generating basic report: {e2}")
    
    except ImportError:
        logger.error("QuantStats not installed. Install with: pip install quantstats")
    except Exception as e:
        logger.error(f"Error generating QuantStats report: {e}")


def prepare_returns_for_quantstats(returns, logger):
    """Prepare returns data for QuantStats report generation."""
    # If returns have a proper datetime index with daily frequency, return as-is
    if len(returns) <= 1:
        return returns
        
    # Ensure the index is sorted
    returns = returns.sort_index()
    
    # If we have intraday data, resample to daily returns
    if (returns.index.duplicated().any() or 
        returns.index.to_series().diff().min() < pd.Timedelta(days=1)):
        logger.info("Resampling intraday data to daily returns")
        # Use last value of each day instead of calculating pct_change again
        daily_returns = returns.resample('D').last().pct_change(fill_method=None).dropna()
        if len(daily_returns) > 1:
            return daily_returns
    
    return returns


def ingest_data(args):
    """Ingest data into the database."""
    # Get logger for CLI
    logger = logging.getLogger('vegas.cli')
    
    try:
        # Initialize data layer
        data_layer = DataLayer(data_dir=args.db_dir, timezone=args.timezone)
        
        if args.file:
            # Check if file exists
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return 1
                
            logger.info(f"Ingesting data from file: {args.file}")
            
            # Load and ingest data
            data_layer.load_data(file_path=args.file)
            logger.info(f"Data ingestion complete")
            
        elif args.directory:
            # Check if directory exists
            if not os.path.exists(args.directory):
                logger.error(f"Directory not found: {args.directory}")
                return 1
                
            logger.info(f"Ingesting data from directory: {args.directory}")
            
            # Set max files if specified
            max_files = args.max_files if args.max_files and args.max_files > 0 else None
            
            # Load and ingest data
            data_layer.load_data(directory=args.directory, file_pattern=args.pattern, max_files=max_files)
            logger.info(f"Data ingestion complete")
            
        else:
            logger.error("No data source specified. Provide --file or --directory")
            return 1
            
        # Print data info
        info = data_layer.get_data_info()
        print("\nData ingestion summary:")
        print(f"Symbols: {info['symbol_count']}")
        if info['start_date'] is not None and info['end_date'] is not None:
            start_date = info['start_date'].date() if hasattr(info['start_date'], 'date') else info['start_date']
            end_date = info['end_date'].date() if hasattr(info['end_date'], 'date') else info['end_date']
            print(f"Date range: {start_date} to {end_date}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


def ingest_ohlcv(args):
    """Ingest OHLCV data into the database."""
    # Get logger for CLI
    logger = logging.getLogger('vegas.cli')
    
    try:
        # Initialize data layer
        data_layer = DataLayer(data_dir=args.db_dir, timezone=args.timezone)
        
        if not data_layer.db_manager:
            logger.error("Database manager initialization failed")
            return 1
            
        if args.file:
            # Ingest a single OHLCV file
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return 1
                
            logger.info(f"Ingesting OHLCV file: {args.file}")
            
            try:
                rows = data_layer.ingest_ohlcv_file(args.file)
                logger.info(f"Ingested {rows} rows from OHLCV file")
                
            except Exception as e:
                logger.error(f"Failed to ingest OHLCV file: {e}")
                return 1
                
        elif args.directory:
            # Ingest OHLCV files from a directory
            if not os.path.exists(args.directory):
                logger.error(f"Directory not found: {args.directory}")
                return 1
                
            logger.info(f"Ingesting OHLCV files from directory: {args.directory}")
            
            try:
                max_files = args.max_files if args.max_files and args.max_files > 0 else None
                rows = data_layer.ingest_ohlcv_directory(args.directory, max_files=max_files)
                logger.info(f"Ingested {rows} rows from OHLCV files")
                
            except Exception as e:
                logger.error(f"Failed to ingest OHLCV directory: {e}")
                return 1
                
        else:
            logger.error("No data source specified. Provide --file or --directory")
            return 1
            
        # Print database status
        print("\nDatabase status after ingestion:")
        db_status_internal(args)
        
        return 0
        
    except Exception as e:
        logger.error(f"OHLCV ingestion failed: {e}")
        return 1


def db_status(args):
    """Display database status."""
    
    return db_status_internal(args)


def db_status_internal(args):
    """Internal implementation of database status command."""
    logger = logging.getLogger('vegas.cli')
    
    # Ensure detailed attribute exists with default value
    if not hasattr(args, 'detailed'):
        args.detailed = False
    
    try:
        # Initialize data layer
        data_layer = DataLayer(data_dir=args.db_dir)
        
        if not data_layer.db_manager:
            logger.error("Database manager initialization failed")
            return 1
            
        # Get database info
        try:
            db_size = data_layer.db_manager.get_database_size()
            dates = data_layer.db_manager.get_available_dates()
            symbols = data_layer.db_manager.get_available_symbols()
            sources = data_layer.db_manager.get_data_sources()
            
            # Print database info
            print("\nDatabase Status:")
            print(f"Database location: {data_layer.db_path}")
            print(f"Database size: {db_size / (1024*1024):.2f} MB")
            
            if dates.empty or pd.isna(dates['start_date'].iloc[0]):
                print("No data available in the database")
                return 0
                
            print(f"Date range: {dates['start_date'].iloc[0].date()} to {dates['end_date'].iloc[0].date()}")
            print(f"Days with data: {dates['day_count'].iloc[0]}")
            print(f"Symbols: {len(symbols)}")
            print(f"Total records: {symbols['record_count'].sum():,}")
            print(f"Data sources: {len(sources)}")
            
            # Print detailed info if requested
            if getattr(args, 'detailed', False):
                if not symbols.empty:
                    print("\nTop 10 symbols by record count:")
                    top_symbols = symbols.nlargest(10, 'record_count')
                    for _, row in top_symbols.iterrows():
                        print(f"  {row['symbol']}: {row['record_count']:,} records")
                        
                if not sources.empty:
                    print("\nData sources:")
                    for _, row in sources.iterrows():
                        added = row['added_date'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['added_date']) else 'Unknown'
                        print(f"  {row['source_name']}: {row['row_count']:,} rows, added on {added}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get database status: {e}")
            return 1
            
    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return 1


def db_query(args):
    """Execute a SQL query on the database."""
    # Get logger for CLI
    logger = logging.getLogger('vegas.cli')
    
    try:
        # Initialize data layer
        data_layer = DataLayer(data_dir=args.db_dir)
        
        if not data_layer.db_manager:
            logger.error("Database manager initialization failed")
            return 1
            
        # Get the query from arguments or file
        query = args.query
        if args.query_file:
            if not os.path.exists(args.query_file):
                logger.error(f"Query file not found: {args.query_file}")
                return 1
                
            with open(args.query_file, 'r') as f:
                query = f.read()
                
        if not query:
            logger.error("No query specified. Use --query or --query-file")
            return 1
            
        # Execute the query
        logger.info("Executing query...")
        try:
            result = data_layer.db_manager.query_to_df(query)
            
            if result.empty:
                print("Query returned no results")
                return 0
                
            # Print results
            if args.output:
                # Save to CSV
                result.to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
            else:
                # Print to console (with limit)
                max_rows = args.limit if args.limit > 0 else len(result)
                pd.set_option('display.max_rows', max_rows)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(result.head(max_rows))
                
                if len(result) > max_rows:
                    print(f"\n(Showing {max_rows} of {len(result)} rows)")
                    
            return 0
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return 1
            
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        return 1


def delete_db(args):
    """Delete the database file."""
    # Get logger for CLI
    logger = logging.getLogger('vegas.cli')
    
    # Calculate the default database path
    db_dir = args.db_dir or "db"
    db_path = os.path.join(db_dir, "vegas.duckdb")
    
    if not os.path.exists(db_path):
        logger.info(f"Database does not exist: {db_path}")
        return 0
        
    # Confirm deletion if not forced
    if not args.force:
        confirm = input(f"Are you sure you want to delete the database at {db_path}? [y/N] ")
        if confirm.lower() not in ['y', 'yes']:
            logger.info("Database deletion cancelled")
            return 0
            
    try:
        # Delete the file
        os.remove(db_path)
        logger.info(f"Database deleted: {db_path}")
        
        # Check for and delete WAL files
        wal_path = db_path + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
            logger.info(f"Database WAL file deleted: {wal_path}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Failed to delete database: {e}")
        return 1


def main():
    """Main entry point for the CLI."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Vegas Backtesting Engine CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    common_parser.add_argument('--db-dir', type=str, default='db', help='Database directory')
    common_parser.add_argument('--timezone', type=str, default='US/Eastern', help='Timezone for data (e.g., UTC, US/Eastern, Europe/London)')
    
    # Run command
    run_parser = subparsers.add_parser('run', parents=[common_parser], help='Run a backtest')
    run_parser.add_argument('strategy_file', type=str, help='Strategy file to run')
    run_parser.add_argument('--data-file', type=str, help='Data file to use')
    run_parser.add_argument('--data-dir', type=str, help='Data directory to use')
    run_parser.add_argument('--file-pattern', type=str, default='*.csv', help='File pattern for data files')
    run_parser.add_argument('--start', dest='start_date', type=parse_date, help='Start date (YYYY-MM-DD)')
    run_parser.add_argument('--end', dest='end_date', type=parse_date, help='End date (YYYY-MM-DD)')
    run_parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    run_parser.add_argument('--output', type=str, help='Output file for equity curve chart')
    run_parser.add_argument('--results-csv', type=str, help='Save results to CSV file')
    run_parser.add_argument('--report', type=str, help='Generate QuantStats report')
    run_parser.add_argument('--benchmark', type=str, help='Benchmark symbol for report')
    # Add new trading hours options
    run_parser.add_argument('--market', type=str, default="US", help='Market name (e.g., NYSE, NASDAQ)')
    run_parser.add_argument('--market-open', type=str, default="09:30", help='Market open time (HH:MM) in 24h format')
    run_parser.add_argument('--market-close', type=str, default="16:00", help='Market close time (HH:MM) in 24h format')
    run_parser.add_argument('--regular-hours-only', action='store_true', help='Only use data from regular market hours')
    run_parser.set_defaults(func=run_backtest)
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', parents=[common_parser], help='Ingest data into the database')
    ingest_source = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_source.add_argument('--file', type=str, help='Data file to ingest')
    ingest_source.add_argument('--directory', type=str, help='Data directory to ingest')
    ingest_parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern for data files')
    ingest_parser.add_argument('--max-files', type=int, help='Maximum number of files to ingest')
    ingest_parser.set_defaults(func=ingest_data)
    
    # Ingest OHLCV command
    ohlcv_parser = subparsers.add_parser('ingest-ohlcv', parents=[common_parser], help='Ingest OHLCV data')
    ohlcv_source = ohlcv_parser.add_mutually_exclusive_group(required=True)
    ohlcv_source.add_argument('--file', type=str, help='OHLCV file to ingest')
    ohlcv_source.add_argument('--directory', type=str, help='Directory with OHLCV files')
    ohlcv_parser.add_argument('--max-files', type=int, help='Maximum number of files to ingest')
    ohlcv_parser.set_defaults(func=ingest_ohlcv)
    
    # DB status command
    status_parser = subparsers.add_parser('db-status', parents=[common_parser], help='Show database status')
    status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
    status_parser.set_defaults(func=db_status)
    
    # DB query command
    query_parser = subparsers.add_parser('db-query', parents=[common_parser], help='Execute SQL query')
    query_parser.add_argument('--query', type=str, help='SQL query to execute')
    query_parser.add_argument('--query-file', type=str, help='File containing SQL query')
    query_parser.add_argument('--output', type=str, help='Output file for query results')
    query_parser.add_argument('--limit', type=int, default=100, help='Maximum rows to display')
    query_parser.set_defaults(func=db_query)
    
    # Delete DB command
    delete_parser = subparsers.add_parser('delete-db', parents=[common_parser], help='Delete the database')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    delete_parser.set_defaults(func=delete_db)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up centralized logging configuration
    # This needs to happen BEFORE any modules/libraries are imported that might set up their own handlers
    log_level = logging.DEBUG if hasattr(args, 'verbose') and args.verbose else logging.INFO
    
    # Remove any existing handlers from the root logger to avoid duplicate messages
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Execute command
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main() 
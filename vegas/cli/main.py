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
    """Load a strategy class from a Python file.
    
    Args:
        file_path: Path to Python file containing a Strategy subclass
        
    Returns:
        Strategy class (not instance)
    """
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
    strategy_classes = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Strategy) and 
            obj != Strategy):
            strategy_classes.append(obj)
    
    if not strategy_classes:
        raise ValueError(f"No Strategy subclass found in {file_path}")
    
    # If multiple strategy classes found, use the first one
    if len(strategy_classes) > 1:
        logging.warning(f"Multiple strategy classes found in {file_path}. Using {strategy_classes[0].__name__}")
    
    return strategy_classes[0]


def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string
        
    Returns:
        datetime object
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def run_backtest(args):
    """Run a backtest with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Load strategy from file
    try:
        strategy_class = load_strategy_from_file(args.strategy_file)
        logger.info(f"Loaded strategy: {strategy_class.__name__}")
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        return 1
    
    # Initialize engine
    engine = BacktestEngine()
    
    # Load data
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
                return 1
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return 1
    
    # Get data info
    data_info = engine.data_layer.get_data_info()
    logger.info(f"Loaded {data_info['row_count']} data points for {data_info['symbol_count']} symbols")
    
    # Set up backtest dates
    start_date = args.start_date or data_info['start_date']
    end_date = args.end_date or data_info['end_date']
    
    logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Create strategy instance
    strategy = strategy_class()
    
    # Run backtest
    try:
        results = engine.run(
            start=start_date,
            end=end_date,
            strategy=strategy,
            initial_capital=args.capital
        )
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return 1
    
    # Print results
    stats = results['stats']
    print("\nBacktest Results:")
    print(f"Total Return: {stats.get('total_return_pct', 0.0):.2f}%")
    
    # Print additional stats if available
    if 'sharpe_ratio' in stats:
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in stats:
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    
    print(f"Number of Trades: {stats.get('num_trades', 0)}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Plot equity curve if requested
    equity_curve = results['equity_curve']
    if not equity_curve.empty:
        # Save equity curve plot if output file is specified
        if args.output:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve['timestamp'], equity_curve['equity'])
            plt.title(f'Portfolio Equity Curve - {strategy_class.__name__}')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.savefig(args.output)
            logger.info(f"Equity curve saved to {args.output}")
        
        # Generate QuantStats report if requested
        if args.report:
            try:
                import quantstats as qs
                import os
                
                # Prepare returns series for QuantStats
                returns = equity_curve.set_index('timestamp')['equity'].pct_change().dropna()
                returns.index = pd.to_datetime(returns.index)
                
                # Check if we have enough data points for a meaningful report
                if len(returns) < 2:
                    logger.warning("Not enough data points for a QuantStats report. Need at least 2 days of data.")
                    return 0
                
                # Get benchmark data if specified
                benchmark = args.benchmark or 'SPY'
                
                # Generate report
                report_path = os.path.abspath(args.report)
                logger.info(f"Generating QuantStats report with benchmark {benchmark}...")
                
                try:
                    # Make sure the returns have a proper datetime index with daily frequency
                    # This helps avoid resampling issues in QuantStats
                    if len(returns) > 1:
                        # Ensure the index is sorted
                        returns = returns.sort_index()
                        
                        # If we have intraday data, resample to daily returns
                        if returns.index.duplicated().any() or returns.index.to_series().diff().min() < pd.Timedelta(days=1):
                            logger.info("Resampling intraday data to daily returns")
                            daily_returns = returns.resample('D').last().pct_change().dropna()
                            if len(daily_returns) > 1:
                                returns = daily_returns
                    
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
                    
                    # Generate the report
                    qs.reports.html(returns, benchmark=benchmark, output=report_path, title=f"{strategy_class.__name__} Performance Report")
                    
                    # Verify the file was created
                    if os.path.exists(report_path):
                        logger.info(f"QuantStats report saved to {report_path}")
                    else:
                        logger.error(f"Failed to create report at {report_path}")
                    
                except Exception as e:
                    logger.warning(f"Error with standard report: {e}. Trying a basic report...")
                    
                    # Try with a simpler report that might avoid the error
                    try:
                        # Create a basic returns tearsheet instead
                        qs.reports.basic(returns, output=report_path, title=f"{strategy_class.__name__} Basic Performance Report")
                        
                        # Verify the file was created
                        if os.path.exists(report_path):
                            logger.info(f"Basic QuantStats report saved to {report_path}")
                        else:
                            logger.error(f"Failed to create basic report at {report_path}")
                    except Exception as e2:
                        logger.error(f"Error generating basic report: {e2}")
            
            except Exception as e:
                logger.error(f"Error generating QuantStats report: {e}")
    
    # Save results to CSV if requested
    if args.results_csv:
        equity_curve.to_csv(args.results_csv, index=False)
        logger.info(f"Results saved to {args.results_csv}")
    
    return 0


def ingest_data(args):
    """Ingest data into the database.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Initialize data layer
    data_layer = DataLayer(args.data_dir)
    
    if not data_layer.use_database:
        logger.error("Database is not available. Please make sure DuckDB and PyArrow are installed.")
        return 1
    
    total_rows = 0
    skipped_files = 0
    
    # Process a single file
    if args.file:
        logger.info(f"Ingesting file: {args.file}")
        try:
            if args.file.endswith('.zst'):
                # Read zstd compressed file
                import zstandard as zstd
                import io
                with open(args.file, 'rb') as f:
                    dctx = zstd.ZstdDecompressor()
                    data_buffer = dctx.decompress(f.read())
                    csv_data = io.StringIO(data_buffer.decode('utf-8'))
                    df = pd.read_csv(csv_data)
            else:
                df = pd.read_csv(args.file)
            
            # Check if this is an OHLCV file (has ts_event column)
            if 'ts_event' in df.columns and 'timestamp' not in df.columns:
                # Rename ts_event to timestamp
                df = df.rename(columns={'ts_event': 'timestamp'})
                
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            source_name = os.path.basename(args.file)
            
            # Check if this file has already been ingested
            existing_sources = data_layer.db_manager.query_to_df(
                "SELECT * FROM data_sources WHERE source_name = ?", 
                (source_name,)
            )
            
            if not existing_sources.empty:
                logger.warning(f"File {source_name} has already been ingested. Skipping to prevent duplicate data.")
                skipped_files += 1
            else:
                rows = data_layer.ingest_to_database(df, source_name)
                total_rows += rows
                logger.info(f"Ingested {rows} rows from {source_name}")
            
        except Exception as e:
            logger.error(f"Error ingesting file {args.file}: {e}")
            return 1
    
    # Process a directory
    elif args.directory:
        logger.info(f"Ingesting files from directory: {args.directory}")
        
        # Find data files
        import glob
        search_path = os.path.join(args.directory, args.pattern)
        files = sorted(glob.glob(search_path))
        
        if not files:
            logger.error(f"No files found in {args.directory} matching pattern {args.pattern}")
            return 1
        
        # Limit the number of files if specified
        if args.max_files:
            files = files[:args.max_files]
        
        logger.info(f"Found {len(files)} files to ingest")
        
        # Process each file
        for file in files:
            try:
                if file.endswith('.zst'):
                    # Read zstd compressed file
                    import zstandard as zstd
                    import io
                    with open(file, 'rb') as f:
                        dctx = zstd.ZstdDecompressor()
                        data_buffer = dctx.decompress(f.read())
                        csv_data = io.StringIO(data_buffer.decode('utf-8'))
                        df = pd.read_csv(csv_data)
                else:
                    df = pd.read_csv(file)
                
                # Check if this is an OHLCV file (has ts_event column)
                if 'ts_event' in df.columns and 'timestamp' not in df.columns:
                    # Rename ts_event to timestamp
                    df = df.rename(columns={'ts_event': 'timestamp'})
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                source_name = os.path.basename(file)
                
                # Check if this file has already been ingested
                existing_sources = data_layer.db_manager.query_to_df(
                    "SELECT * FROM data_sources WHERE source_name = ?", 
                    (source_name,)
                )
                
                if not existing_sources.empty:
                    logger.warning(f"File {source_name} has already been ingested. Skipping to prevent duplicate data.")
                    skipped_files += 1
                else:
                    rows = data_layer.ingest_to_database(df, source_name)
                    total_rows += rows
                    logger.info(f"Ingested {rows} rows from {source_name}")
                
            except Exception as e:
                logger.error(f"Error ingesting file {file}: {e}")
    
    else:
        logger.error("No ingestion source specified. Use --file or --directory.")
        return 1
    
    # Display summary
    print(f"\nIngestion completed: {total_rows} total rows ingested")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files that were already ingested")
    
    # Display database status
    db_info = data_layer.get_data_info()
    print("\nDatabase Status:")
    print(f"Total rows: {db_info['row_count']}")
    print(f"Total symbols: {db_info['symbol_count']}")
    if db_info['start_date'] and db_info['end_date']:
        print(f"Date range: {db_info['start_date'].date()} to {db_info['end_date'].date()}")
    print(f"Unique days: {db_info['day_count']}")
    print(f"Database size: {db_info['database_size_mb']} MB")
    
    return 0


def ingest_ohlcv(args):
    """Ingest OHLCV files into the database.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Initialize data layer with db directory
    data_layer = DataLayer("db")
    
    if not data_layer.use_database:
        logger.error("Database is not available. Please make sure DuckDB and PyArrow are installed.")
        return 1
    
    total_rows = 0
    skipped_files = 0
    
    # Process a single file
    if args.file:
        logger.info(f"Ingesting OHLCV file: {args.file}")
        try:
            # Use the database manager's method for ingesting OHLCV files
            rows = data_layer.db_manager.ingest_ohlcv_file(args.file)
            if rows > 0:
                total_rows += rows
                logger.info(f"Ingested {rows} rows from {os.path.basename(args.file)}")
            else:
                skipped_files += 1
                logger.info(f"Skipped {os.path.basename(args.file)} (already ingested)")
        except Exception as e:
            logger.error(f"Error ingesting OHLCV file {args.file}: {e}")
            return 1
    
    # Process all files in a directory
    elif args.directory:
        logger.info(f"Ingesting OHLCV files from directory: {args.directory}")
        
        try:
            # Use the database manager's method for ingesting OHLCV directories
            total_rows = data_layer.db_manager.ingest_ohlcv_directory(
                args.directory, 
                pattern="*.ohlcv-1h.csv.zst", 
                max_files=args.max_files
            )
            
            # Get the number of skipped files from logs or database
            skipped_files_df = data_layer.db_manager.query_to_df("""
                SELECT COUNT(*) as skipped_count 
                FROM data_sources 
                WHERE source_name LIKE '%.ohlcv-1h.csv.zst' 
                  AND row_count = 0
            """)
            if not skipped_files_df.empty:
                skipped_files = skipped_files_df['skipped_count'].iloc[0]
            
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Error ingesting OHLCV files from {args.directory}: {e}")
            return 1
    
    else:
        logger.error("No ingestion source specified. Use --file or --directory.")
        return 1
    
    # Display summary
    print(f"\nOHLCV Ingestion completed: {total_rows} total rows ingested")
    if skipped_files > 0:
        print(f"Skipped {skipped_files} files that were already ingested")
    
    # Display database status
    db_info = data_layer.get_data_info()
    print("\nDatabase Status:")
    print(f"Total rows: {db_info['row_count']}")
    print(f"Total symbols: {db_info['symbol_count']}")
    if db_info['start_date'] and db_info['end_date']:
        print(f"Date range: {db_info['start_date'].date()} to {db_info['end_date'].date()}")
    print(f"Unique days: {db_info['day_count']}")
    print(f"Database size: {db_info['database_size_mb']} MB")
    
    return 0


def db_status(args):
    """Display database status.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Initialize data layer
    data_layer = DataLayer(args.data_dir)
    
    if not data_layer.use_database:
        logger.error("Database is not available. Please make sure DuckDB and PyArrow are installed.")
        return 1
    
    try:
        # Display database status
        db_info = data_layer.get_data_info()
        print("\nDatabase Status:")
        print(f"Total rows: {db_info['row_count']}")
        print(f"Total symbols: {db_info['symbol_count']}")
        if db_info['start_date'] and db_info['end_date']:
            print(f"Date range: {db_info['start_date'].date()} to {db_info['end_date'].date()}")
        print(f"Unique days: {db_info['day_count']}")
        print(f"Database size: {db_info['database_size_mb']} MB")
        
        if args.detailed:
            # Get data sources
            sources = data_layer.db_manager.get_data_sources()
            
            if not sources.empty:
                print("\nData Sources:")
                for _, row in sources.iterrows():
                    print(f"- {row['source_name']}: {row['row_count']} rows "
                          f"({row['start_date'].date()} to {row['end_date'].date()})")
            
            # Get symbol statistics
            symbols = data_layer.db_manager.get_available_symbols()
            
            if not symbols.empty and args.show_symbols:
                print(f"\nAvailable Symbols ({len(symbols)}):")
                
                # Determine how many symbols to show
                limit = min(args.limit, len(symbols)) if args.limit else len(symbols)
                
                for i, (_, row) in enumerate(symbols.iterrows()):
                    if i >= limit:
                        print(f"... and {len(symbols) - limit} more symbols")
                        break
                    print(f"- {row['symbol']}: {row['record_count']} records "
                          f"({row['first_date'].date()} to {row['last_date'].date()})")
        
    except Exception as e:
        logger.error(f"Error accessing database: {e}")
        return 1
    
    return 0


def db_query(args):
    """Run a SQL query on the database.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Initialize data layer
    data_layer = DataLayer(args.data_dir)
    
    if not data_layer.use_database:
        logger.error("Database is not available. Please make sure DuckDB and PyArrow are installed.")
        return 1
    
    try:
        # Get the SQL query
        sql_query = args.query
        
        if args.file:
            # Read query from file
            with open(args.file, 'r') as f:
                sql_query = f.read()
        
        if not sql_query:
            logger.error("No SQL query provided.")
            return 1
        
        # Run the query
        logger.info(f"Executing SQL query: {sql_query}")
        result = data_layer.db_manager.query_to_df(sql_query)
        
        # Output the result
        if result.empty:
            print("Query returned no results.")
        else:
            if args.output:
                # Write to file in the specified format
                ext = args.output.lower().split('.')[-1]
                
                if ext == 'csv':
                    result.to_csv(args.output, index=False)
                elif ext == 'parquet':
                    result.to_parquet(args.output, index=False)
                elif ext == 'json':
                    result.to_json(args.output, orient='records', date_format='iso')
                else:
                    # Default to CSV
                    result.to_csv(args.output, index=False)
                
                print(f"Query results saved to {args.output}")
            else:
                # Display results to console
                pd.set_option('display.max_rows', 100)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print("\nQuery Results:")
                print(result)
                print(f"\n{len(result)} rows returned")
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return 1
    
    return 0


def delete_db(args):
    """Delete the database and all parquet files.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('vegas.cli')
    
    # Confirm deletion if not forced
    if not args.force:
        confirm = input("This will delete all database files. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Database deletion cancelled.")
            return 0
    
    # Define paths to delete
    db_path = os.path.join(args.data_dir, "vegas.duckdb")
    partitioned_dir = os.path.join(args.data_dir, "partitioned")
    
    # Delete the database file
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info(f"Deleted database file: {db_path}")
        except Exception as e:
            logger.error(f"Error deleting database file {db_path}: {e}")
            return 1
    else:
        logger.info(f"Database file not found: {db_path}")
    
    # Delete all parquet files
    if os.path.exists(partitioned_dir):
        try:
            import shutil
            shutil.rmtree(partitioned_dir)
            logger.info(f"Deleted all parquet files in: {partitioned_dir}")
        except Exception as e:
            logger.error(f"Error deleting parquet files in {partitioned_dir}: {e}")
            return 1
    else:
        logger.info(f"Partitioned directory not found: {partitioned_dir}")
    
    print("Database and all parquet files have been deleted successfully.")
    return 0


def main():
    """Main entry point for the Vegas CLI."""
    parser = argparse.ArgumentParser(
        description="Vegas Backtesting Engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a backtest")
    run_parser.add_argument("strategy_file", help="Path to Python file containing a Strategy subclass")
    
    # Data source options
    data_group = run_parser.add_mutually_exclusive_group(required=False)
    data_group.add_argument("--data-file", help="Path to a single data file (optional if data is already ingested)")
    data_group.add_argument("--data-dir", help="Directory containing data files (optional if data is already ingested)")
    
    # Date range options
    run_parser.add_argument("--start", dest="start_date", type=parse_date, help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end", dest="end_date", type=parse_date, help="End date (YYYY-MM-DD)")
    
    # Other options
    run_parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    run_parser.add_argument("--output", help="Output file for equity curve plot")
    run_parser.add_argument("--results-csv", help="Output file for results CSV")
    run_parser.add_argument("--file-pattern", default="*.csv*", help="Pattern for matching data files")
    run_parser.add_argument("--report", help="Generate a QuantStats HTML report and save to the specified file")
    run_parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol for the QuantStats report (default: SPY)")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into the database")
    ingest_source = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_source.add_argument("--file", help="Path to a single data file to ingest")
    ingest_source.add_argument("--directory", help="Directory containing data files to ingest")
    
    ingest_parser.add_argument("--pattern", default="*.csv*", help="Pattern for matching data files")
    ingest_parser.add_argument("--max-files", type=int, help="Maximum number of files to ingest")
    ingest_parser.add_argument("--data-dir", default="db", help="Base data directory")
    ingest_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Ingest OHLCV command
    ohlcv_parser = subparsers.add_parser("ingest-ohlcv", help="Ingest OHLCV files into the database")
    ohlcv_source = ohlcv_parser.add_mutually_exclusive_group(required=True)
    ohlcv_source.add_argument("--file", help="Path to a single OHLCV file to ingest")
    ohlcv_source.add_argument("--directory", help="Directory containing OHLCV files to ingest")
    
    ohlcv_parser.add_argument("--max-files", type=int, help="Maximum number of files to ingest")
    ohlcv_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # DB Status command
    status_parser = subparsers.add_parser("db-status", help="Display database status")
    status_parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    status_parser.add_argument("--show-symbols", action="store_true", help="Show list of available symbols")
    status_parser.add_argument("--limit", type=int, help="Limit number of symbols to show")
    status_parser.add_argument("--data-dir", default="db", help="Base data directory")
    status_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # DB Query command
    query_parser = subparsers.add_parser("db-query", help="Run a SQL query on the database")
    query_source = query_parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--query", help="SQL query to execute")
    query_source.add_argument("--file", help="File containing SQL query")
    
    query_parser.add_argument("--output", help="Output file for query results")
    query_parser.add_argument("--data-dir", default="db", help="Base data directory")
    query_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Delete DB command
    delete_parser = subparsers.add_parser("delete-db", help="Delete the database and all parquet files")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Force deletion without confirmation")
    delete_parser.add_argument("--data-dir", default="db", help="Base data directory")
    delete_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "run":
        return run_backtest(args)
    elif args.command == "ingest":
        return ingest_data(args)
    elif args.command == "ingest-ohlcv":
        return ingest_ohlcv(args)
    elif args.command == "db-status":
        return db_status(args)
    elif args.command == "db-query":
        return db_query(args)
    elif args.command == "delete-db":
        return delete_db(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
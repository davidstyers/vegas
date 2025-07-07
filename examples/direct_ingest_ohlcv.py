#!/usr/bin/env python3
"""Script to directly ingest OHLCV files into the database.

This script bypasses the CLI and directly uses the DatabaseManager to ingest OHLCV files.
"""

import os
import sys
import glob
import pandas as pd
import subprocess
import tempfile
import logging
from pathlib import Path

# Add the parent directory to the path so we can import vegas modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vegas.database import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('direct_ingest')

def read_ohlcv_file(file_path):
    """Read an OHLCV file using zstdcat and pandas.
    
    Args:
        file_path: Path to the OHLCV file
        
    Returns:
        DataFrame with the file contents
    """
    logger.info(f"Reading OHLCV file: {file_path}")
    
    # Create a temporary file to store the decompressed data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use zstdcat to decompress the file
        cmd = f"zstdcat {file_path} > {temp_path}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Read the decompressed file
        df = pd.read_csv(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        # Process the data
        logger.info(f"Read {len(df)} rows from {file_path}")
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'ts_event': 'timestamp',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'open': 'open',
            'volume': 'volume',
            'symbol': 'symbol'
        })
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading OHLCV file {file_path}: {e}")
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

def main():
    """Main function to ingest OHLCV files."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Directly ingest OHLCV files into the database")
    parser.add_argument("--file", help="Path to a single OHLCV file to ingest")
    parser.add_argument("--directory", help="Directory containing OHLCV files to ingest")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to ingest")
    parser.add_argument("--db-dir", default="db", help="Directory for the database")
    args = parser.parse_args()
    
    # Create the database directory if it doesn't exist
    os.makedirs(args.db_dir, exist_ok=True)
    
    # Initialize the database manager
    db_path = os.path.join(args.db_dir, "vegas.duckdb")
    db_manager = DatabaseManager(db_path, args.db_dir)
    
    total_rows = 0
    
    # Process a single file
    if args.file:
        df = read_ohlcv_file(args.file)
        if df is not None and not df.empty:
            source_name = os.path.basename(args.file)
            rows = db_manager.ingest_data(df, source_name)
            total_rows += rows
            logger.info(f"Ingested {rows} rows from {source_name}")
    
    # Process all files in a directory
    elif args.directory:
        # Find all OHLCV files
        search_path = os.path.join(args.directory, "*.ohlcv-1h.csv.zst")
        files = sorted(glob.glob(search_path))
        
        if not files:
            logger.error(f"No OHLCV files found in {args.directory}")
            return 1
        
        # Limit the number of files if specified
        if args.max_files:
            files = files[:args.max_files]
        
        logger.info(f"Found {len(files)} OHLCV files to ingest")
        
        # Process each file
        for file in files:
            df = read_ohlcv_file(file)
            if df is not None and not df.empty:
                source_name = os.path.basename(file)
                rows = db_manager.ingest_data(df, source_name)
                total_rows += rows
                logger.info(f"Ingested {rows} rows from {source_name}")
    
    else:
        logger.error("No ingestion source specified. Use --file or --directory.")
        return 1
    
    # Display summary
    print(f"\nIngestion completed: {total_rows} total rows ingested")
    
    # Display database status
    try:
        dates_df = db_manager.get_available_dates()
        symbols_df = db_manager.get_available_symbols()
        
        print("\nDatabase Status:")
        print(f"Total rows: {symbols_df['record_count'].sum() if not symbols_df.empty else 0}")
        print(f"Total symbols: {len(symbols_df) if not symbols_df.empty else 0}")
        if not dates_df.empty and dates_df['start_date'].iloc[0] and dates_df['end_date'].iloc[0]:
            print(f"Date range: {dates_df['start_date'].iloc[0].date()} to {dates_df['end_date'].iloc[0].date()}")
        print(f"Unique days: {dates_df['day_count'].iloc[0] if not dates_df.empty else 0}")
        print(f"Database size: {round(db_manager.get_database_size() / (1024 * 1024), 2)} MB")
    except Exception as e:
        logger.error(f"Error getting database status: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
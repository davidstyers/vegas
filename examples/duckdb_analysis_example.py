#!/usr/bin/env python3
"""Example script demonstrating how to use DuckDB for market data analysis.

This script demonstrates how to:
1. Connect to the Vegas DuckDB database
2. Run SQL queries to analyze market data
3. Create simple visualizations of the results
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import argparse
from pathlib import Path

# Set up matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['figure.dpi'] = 100


def connect_to_database(db_path="db/vegas.duckdb"):
    """Connect to the Vegas DuckDB database.
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        DuckDB connection
    """
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        print("Please ingest data first using: vegas ingest-ohlcv --directory path/to/data")
        return None
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Install and load parquet extension
        conn.execute("INSTALL parquet; LOAD parquet;")
        print(f"Connected to DuckDB database at {db_path}")
        
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def run_sql_query(conn, query):
    """Run a SQL query on the DuckDB connection.
    
    Args:
        conn: DuckDB connection
        query: SQL query to run
        
    Returns:
        Pandas DataFrame with results
    """
    try:
        result = conn.execute(query)
        df = result.fetchdf()
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()


def analyze_market_data(conn, output_dir="results"):
    """Run various analyses on market data.
    
    Args:
        conn: DuckDB connection
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis 1: Daily volume by symbol
    print("\nAnalyzing daily volume by symbol...")
    volume_query = """
    SELECT 
        symbol, 
        DATE_TRUNC('day', timestamp) as date, 
        SUM(volume) as daily_volume
    FROM market_data
    GROUP BY symbol, date
    ORDER BY daily_volume DESC
    LIMIT 1000
    """
    volume_df = run_sql_query(conn, volume_query)
    
    if not volume_df.empty:
        # Get top 5 symbols by volume
        top_symbols = volume_df.groupby('symbol')['daily_volume'].sum().sort_values(ascending=False).head(5).index.tolist()
        
        # Filter for top symbols
        top_volume_df = volume_df[volume_df['symbol'].isin(top_symbols)]
        
        # Pivot for plotting
        pivot_df = top_volume_df.pivot(index='date', columns='symbol', values='daily_volume')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_df.plot(ax=ax)
        plt.title('Daily Trading Volume for Top 5 Symbols')
        plt.ylabel('Volume')
        plt.xlabel('Date')
        plt.legend(title='Symbol')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'daily_volume.png'))
        print(f"Saved daily volume chart to {output_dir}/daily_volume.png")
    
    # Analysis 2: Price volatility
    print("\nAnalyzing price volatility...")
    volatility_query = """
    SELECT 
        symbol, 
        DATE_TRUNC('day', timestamp) as date,
        (MAX(high) - MIN(low)) / AVG(close) * 100 as daily_volatility
    FROM market_data
    GROUP BY symbol, date
    HAVING COUNT(*) > 10
    ORDER BY daily_volatility DESC
    LIMIT 1000
    """
    volatility_df = run_sql_query(conn, volatility_query)
    
    if not volatility_df.empty:
        # Get top 5 volatile symbols
        volatile_symbols = volatility_df.groupby('symbol')['daily_volatility'].mean().sort_values(ascending=False).head(5).index.tolist()
        
        # Filter for volatile symbols
        vol_df = volatility_df[volatility_df['symbol'].isin(volatile_symbols)]
        
        # Pivot for plotting
        pivot_df = vol_df.pivot(index='date', columns='symbol', values='daily_volatility')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_df.plot(ax=ax)
        plt.title('Daily Price Volatility for Most Volatile Symbols')
        plt.ylabel('Volatility (%)')
        plt.xlabel('Date')
        plt.legend(title='Symbol')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'volatility.png'))
        print(f"Saved volatility chart to {output_dir}/volatility.png")
    
    # Analysis 3: Correlation between symbols
    print("\nAnalyzing correlation between symbols...")
    corr_query = """
    WITH daily_returns AS (
        SELECT 
            symbol,
            DATE_TRUNC('day', timestamp) as date,
            (MAX(close) - MIN(open)) / MIN(open) * 100 as daily_return
        FROM market_data
        GROUP BY symbol, date
        HAVING COUNT(*) > 10
    )
    SELECT *
    FROM daily_returns
    WHERE symbol IN (
        SELECT symbol
        FROM market_data
        GROUP BY symbol
        ORDER BY COUNT(*) DESC
        LIMIT 10
    )
    """
    returns_df = run_sql_query(conn, corr_query)
    
    if not returns_df.empty:
        # Pivot for correlation
        pivot_df = returns_df.pivot(index='date', columns='symbol', values='daily_return')
        
        # Calculate correlation
        corr_df = pivot_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation of Daily Returns between Top Symbols')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation.png'))
        print(f"Saved correlation heatmap to {output_dir}/correlation.png")


def main():
    parser = argparse.ArgumentParser(description="DuckDB Market Data Analysis Example")
    parser.add_argument("--db-path", default="db/vegas.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="results", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_to_database(args.db_path)
    if not conn:
        return 1
    
    # Check if market_data view exists
    try:
        result = conn.execute("SELECT COUNT(*) FROM market_data")
        count = result.fetchone()[0]
        print(f"Found {count} rows in market_data view")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've ingested data into the database first")
        return 1
    
    # Run analyses
    analyze_market_data(conn, args.output_dir)
    
    # Close connection
    conn.close()
    
    print("\nAnalysis complete. Check the output directory for visualizations.")
    return 0


if __name__ == "__main__":
    exit(main()) 
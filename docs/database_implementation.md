# DuckDB and Parquet Database Implementation

## Overview

We've implemented a DuckDB and Parquet-based database system for the Vegas backtesting engine to efficiently store and query market data. This document provides an overview of the implementation.

## Components Added

1. **Database Module**
   - `DatabaseManager`: Core class for managing the DuckDB database
   - `ParquetManager`: Helper class for reading/writing Parquet files

2. **CLI Commands**
   - `ingest`: Command for ingesting data into the database
   - `db-status`: Command for checking database status
   - `db-query`: Command for running SQL queries on the database

3. **DataLayer Integration**
   - Updated the DataLayer class to use the database for data retrieval
   - Added fallback to in-memory data when the database is not available

4. **Examples and Documentation**
   - Added examples for using the database system
   - Added documentation on the database architecture and usage

## Architecture

### Database Structure

- **market_data**: SQL view over Parquet files for querying market data
- **symbols**: Table tracking available symbols
- **data_sources**: Table tracking ingested data sources
- **Partitioning**: Data is partitioned by year and symbol in Parquet files

### Data Flow

1. Data is ingested from CSV/zstd files
2. Data is converted to Parquet format and stored in partitioned directories
3. DuckDB reads Parquet files directly when queried
4. Results are returned as pandas DataFrames

### Fallback Mechanism

The system is designed to gracefully fall back to in-memory pandas when:
- DuckDB or PyArrow dependencies are not available
- The database connection fails
- No data is found in the database for a specific query

## Dependencies

- **DuckDB**: Fast in-process analytical database
- **PyArrow**: Python library for Apache Arrow and Parquet format
- **pandas**: For DataFrame operations

## Benefits

1. **Performance**
   - Reduced memory usage through columnar storage
   - Fast SQL queries on large datasets
   - Efficient filtering by date range and symbols

2. **Storage Efficiency**
   - Parquet compression (Snappy) reduces disk space usage
   - Partitioning improves query performance
   - Columnar storage optimizes for analytical queries

3. **User Experience**
   - SQL interface for data exploration
   - CLI commands for database management
   - Seamless integration with existing backtesting workflow

## Testing

The database functionality has been tested with:
- Unit tests for the `DatabaseManager` and `ParquetManager` classes
- Integration tests with the `DataLayer` class
- Manual testing with example scripts

## Future Improvements

Potential future enhancements to the database system:
- Add support for more market data sources
- Implement incremental ingestion to avoid duplicate data
- Add more advanced SQL query examples for market analysis
- Optimize partitioning strategy based on usage patterns
- Add database maintenance commands (vacuum, optimize)
- Support for multi-user access in a shared environment

## Example Use Cases

1. **Large-scale backtesting**
   - Efficiently store and query years of market data
   - Run backtests on specific symbols and date ranges without loading all data

2. **Market data analysis**
   - Use SQL to analyze market trends and patterns
   - Create custom datasets for machine learning models
   
3. **Portfolio optimization**
   - Query historical data for multiple assets
   - Calculate correlations and other statistical measures 
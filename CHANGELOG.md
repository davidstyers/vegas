# Changelog

## [Unreleased] - 2025-09-07

### Added
- Enhanced data ingestion and CLI functionality for TBBO tick data
- Signal generation capabilities to BacktestEngine and Strategy classes
- Alpha analytics module with decision tracking and metrics
- Data transformation framework with frequency management
- Tick bars and OHLCV resampling capabilities
- Tabulate utility for data formatting
- Data portal for enhanced data access patterns

### Changed
- Refactored strategy context handling and removed deprecated files
- Enhanced performance analytics and reporting capabilities
- Streamlined pipeline scheduling in BacktestEngine
- Improved CLI interface with better calendar selection support

### Fixed
- Removed deprecated database files and cleaned up repository structure

## [0.0.2] - 2025-08-19

### Added
- Unified calendar system for timestamp filtering
- Enhanced performance analytics and reporting capabilities
- Hyperopt integration for strategy optimization
- IBKR (Interactive Brokers) support and broker adapter functionality
- Live trading capabilities with market data feeds
- Simulated broker adapters for testing
- Pipeline system for data processing and factor generation
- Advanced filtering and statistical factors
- Comprehensive test suite with integration tests
- Performance benchmarking tools
- Calendar-based market hours handling

### Changed
- Enhanced BacktestEngine to utilize unified calendar system
- Improved CLI interface to support calendar selection for backtesting
- Refactored various components for better architecture alignment
- Updated documentation and formatting across multiple files
- Enhanced broker adapter functionality
- Improved pipeline functionality and dependencies
- Streamlined repository structure by removing obsolete files

### Fixed
- Enhanced timezone handling in database conversions
- Improved error handling in database connections
- Fixed pipeline return types and data handling
- Repository cleanup by removing compiled Python files

## [1.0.0] - 2023-07-15

### Added
- Added high-performance data processing with Polars
- Added migration guide for transitioning from pandas to polars
- Added improved DataFrame schema handling for empty DataFrames
- Added database-level timezone conversion for improved performance
- Added database-level filtering for regular trading hours

### Changed
- Migrated core data layer from pandas to polars
- Optimized database operations to use polars
- Updated backtest engine for compatibility with polars
- Updated examples and documentation to use polars API
- Improved timezone handling using native polars functionality
- Restructured groupby operations to match polars API
- Optimized memory usage in data ingestion pipeline

### Fixed
- Fixed performance bottlenecks in large dataset processing
- Fixed timezone conversion issues in timestamp handling
- Improved error handling in database connections

## [0.9.0] - 2023-05-01

### Added
- Initial release with pandas-based implementation
- Event-driven backtesting engine
- DuckDB and Parquet storage
- Market hours handling
- Timezone support
- CLI interface

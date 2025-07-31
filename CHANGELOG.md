# Changelog

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
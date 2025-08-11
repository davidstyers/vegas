# Vegas Codebase Test Plan

## 1. Introduction

This document outlines a comprehensive testing strategy for the `vegas` codebase. The current test coverage is critically low, exposing the project to significant risk. This plan prioritizes testing efforts based on business impact and component criticality, with the ultimate goal of achieving 100% coverage of all meaningful code.

## 2. Testing Priorities

The testing effort will be divided into three priority levels:

### Priority 1 (Critical)

These components are the core of the backtesting engine. Their failure would be catastrophic to the system's reliability and correctness.

*   **`vegas/engine`**: The backtesting engine itself.
*   **`vegas/broker`**: The order execution and management system.
*   **`vegas/portfolio`**: The position and performance tracking system.

### Priority 2 (High)

These components are essential for supporting the core engine. Their failure would significantly impair the system's functionality.

*   **`vegas/data`**: The data loading, ingestion, and querying layer.
*   **`vegas/analytics`**: The performance analytics and reporting layer.

### Priority 3 (Medium)

These components are lower-risk utilities and auxiliary modules.

*   **`vegas/pipeline`**: The alpha factor research and computation engine.
*   **`vegas/cli`**: The command-line interface.
*   **`vegas/utils`**: Shared utilities.
*   **`vegas/calendars`**: Trading calendar implementations.

## 3. Test Scope and Types

### 3.1. Priority 1: Core Components

#### 3.1.1. `vegas/engine`

*   **Unit Tests:**
    *   Test `BacktestEngine` initialization with various configurations (e.g., different timezones, data directories).
    *   Test `set_trading_hours` and `ignore_extended_hours` functionality.
    *   Test pipeline attachment and retrieval (`attach_pipeline`, `pipeline_output`).
    *   Test universe discovery logic (`_discover_universe`).
    *   Test `_is_regular_market_hours` with various timestamps.
    *   Test `load_data` with different sources (file, directory, database).
    *   Test the main `run` method's orchestration of the backtest loop.
*   **Integration Tests:**
    *   Run a full backtest with a simple strategy and assert on the final results (e.g., equity, number of trades).
    *   Test the interaction between the engine, broker, and portfolio during a backtest.
    *   Test the engine's handling of data loading failures.
    *   Test the engine's handling of strategies that raise exceptions.

#### 3.1.2. `vegas/broker`

*   **Unit Tests:**
    *   Test `Order` creation with all supported order types and parameters.
    *   Test `Transaction` and `Position` classes.
    *   Test `Broker` initialization.
    *   Test `place_order` with various signal types and bracket orders.
    *   Test `execute_orders` with different market data scenarios (e.g., gaps, no data).
    *   Test slippage and commission models.
    *   Test order cancellation (`cancel_order`).
*   **Integration Tests:**
    *   Test the full lifecycle of an order, from placement to execution to transaction recording.
    *   Test the interaction between the broker and the data portal during order execution.
    *   Test the broker's handling of insufficient cash or buying power.

#### 3.1.3. `vegas/portfolio`

*   **Unit Tests:**
    *   Test `Portfolio` initialization.
    *   Test `update_from_transactions` with various transaction types (buys, sells, shorts).
    *   Test margin and buying power calculations (`_recompute_short_margin_and_buying_power`).
    *   Test `get_stats` and other metric calculation methods.
    *   Test `build_positions_ledger` with open and closed positions.
*   **Integration Tests:**
    *   Test the portfolio's state changes over a multi-day backtest.
    *   Test the interaction between the portfolio and the data portal for price updates.
    *   Test the portfolio's handling of short selling and margin calls.

### 3.2. Priority 2: Supporting Modules

#### 3.2.1. `vegas/data`

*   **Unit Tests:**
    *   Test `DataLayer` initialization in both normal and test modes.
    *   Test `load_data` with various file formats (CSV, compressed CSV) and sources.
    *   Test database ingestion and querying (`ingest_to_database`, `get_data_for_backtest`).
    *   Test timezone conversions (`_convert_timestamp_timezone`).
    *   Test `history` method with different parameters.
*   **Integration Tests:**
    *   Test the full data loading and querying pipeline, from raw files to database to backtest.
    *   Test the data layer's handling of missing or corrupted data files.

#### 3.2.2. `vegas/analytics`

*   **Unit Tests:**
    *   Test `Results` class creation and serialization.
    *   Test `Analytics.calculate_stats` with mock portfolio and broker data.
    *   Mock the `quantstats` library to test the plotting and report generation methods without relying on the external library's correctness.
*   **Integration Tests:**
    *   Test the full analytics pipeline, from backtest results to a generated tearsheet.

### 3.3. Priority 3: Auxiliary Components

#### 3.3.1. `vegas/pipeline`

*   **Unit Tests:**
    *   Test `PipelineEngine` initialization.
    *   Test `run_pipeline` with various factors, filters, and classifiers.
    *   Test `_get_max_window_length`.
*   **Integration Tests:**
    *   Test the pipeline engine's interaction with the data portal.
    *   Test a pipeline with custom factors and filters.

#### 3.3.2. `vegas/cli`

*   **Unit Tests:**
    *   Use `unittest.mock` to test the argument parsing and function calls for each CLI command (`run`, `ingest`, `db-status`, etc.).
*   **Integration Tests:**
    *   Run the CLI with a simple backtest and assert on the output files and console logs.
    *   Test the CLI's error handling for invalid arguments and file paths.

## 4. Exclusions

The following components will be excluded from the 100% test coverage target:

*   **Third-Party Library Stubs:** Mocks and stubs for external libraries (e.g., `quantstats`) are not part of the application's logic and will not be tested directly.
*   **Simple Data Structures:** Simple `dataclass` objects with no custom logic (e.g., `vegas.strategy.Signal`) do not require dedicated tests.
*   **Boilerplate Code:** Code that is generated automatically or is part of a standard framework (e.g., `__main__` blocks) will be excluded.

## 5. Tooling and Frameworks

*   **Test Runner:** `pytest`
*   **Mocking:** `unittest.mock`
*   **Code Coverage:** `pytest-cov`

This plan provides a clear roadmap for achieving comprehensive test coverage for the `vegas` codebase. By following these priorities and scopes, we can significantly improve the project's stability, reliability, and maintainability.
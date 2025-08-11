# Architectural Plan: Redesign of the Backtesting Engine's Data Layer

## 1. Introduction

This document outlines the architectural redesign of the data layer for the Vegas backtesting engine. The primary goal is to introduce a centralized, in-memory data cache service, named `DataPortal`, to streamline data access, improve performance, and simplify the overall architecture. This plan will serve as a guide for the implementation phase.

## 2. Current Architecture and its Limitations

The existing data layer, implemented in `vegas/data/data_layer.py`, is responsible for loading, caching, and providing access to historical market data. While functional, it has several limitations:

- **Decentralized Caching**: Caching logic is spread across multiple methods within the `DataLayer` class, making it difficult to manage and reason about.
- **Redundant Data Loading**: The current implementation can lead to redundant data loading and processing, impacting performance.
- **Complex Data Access Patterns**: The `DataPortal` currently acts as a thin wrapper around the `DataLayer`, leading to complex data access patterns and tight coupling between components.

## 3. Proposed Architecture: The Centralized DataPortal

To address these limitations, we will refactor the `DataPortal` to act as a centralized, in-memory data cache. This new design will simplify data access, improve performance, and provide a clear separation of concerns.

### 3.1. The `DataPortal` Class

The redesigned `DataPortal` class will be responsible for the following:

- **Pre-loading Data**: At the start of a backtest, the `DataPortal` will pre-load all necessary data for the specified time range and symbols into an in-memory cache.
- **Providing Data Slices**: The `DataPortal` will expose a simple API for accessing data slices for a given timestamp.
- **Managing the Cache**: The `DataPortal` will manage the in-memory cache, ensuring that data is loaded only once per backtest.

### 3.2. Public API of the `DataPortal`

The `DataPortal` class will have the following public API:

```python
class DataPortal:
    def __init__(self, data_layer: DataLayer):
        """
        Initializes the DataPortal.

        Args:
            data_layer: An instance of the DataLayer class.
        """
        ...

    def load_data(self, start_date: datetime, end_date: datetime, symbols: list[str]):
        """
        Pre-loads all necessary data for the given date range and symbols
        into an in-memory cache.

        Args:
            start_date: The start date of the backtest.
            end_date: The end date of the backtest.
            symbols: A list of symbols to load data for.
        """
        ...

    def get_spot_value(self, asset: str, field: str, dt: datetime):
        """
        Look up a single value at an arbitrary timestamp.
        """
        ...

    def get_slice(self, timestamp: datetime) -> pl.DataFrame:
        """
        Returns a DataFrame containing the data for the given timestamp.

        Args:
            timestamp: The timestamp to get data for.

        Returns:
            A Polars DataFrame containing the data for the given timestamp.
        """
        ...
```

### 3.3. Integration with the `BacktestEngine`

The `DataPortal` will be instantiated and managed within the `BacktestEngine.run()` method. The following changes will be made to `vegas/engine/engine.py`:

1.  **Instantiate the `DataPortal`**: In the `run()` method, a `DataPortal` instance will be created and passed to the `Strategy`, `Portfolio`, and `Broker` instances.
2.  **Pre-load Data**: Before the backtest loop starts, the `DataPortal.load_data()` method will be called to pre-load all necessary data.
3.  **Replace `data.history` Calls**: All calls to `data.history` will be replaced with calls to the `DataPortal.get_slice()` method.

### 3.4. Dependency Injection

To decouple the `Strategy`, `Portfolio`, and `Broker` from the `DataLayer`, we will use dependency injection. The `DataPortal` instance will be passed to these components through their constructors. This will allow us to easily replace the `DataPortal` with a different implementation in the future, if needed.

### 3.5. Removing Redundant Caching Logic

Once the new `DataPortal` is in place, we will remove the redundant caching logic from the `DataLayer` class. This will simplify the `DataLayer` and make it easier to maintain.

## 4. Implementation Plan

The implementation will be carried out in the following steps:

1.  **Refactor the `DataPortal` Class**: Implement the new `DataPortal` class with the specified public API.
2.  **Integrate the `DataPortal` into the `BacktestEngine`**: Modify the `BacktestEngine.run()` method to instantiate and manage the `DataPortal`.
3.  **Update the `Strategy`, `Portfolio`, and `Broker`**: Update these components to use the `DataPortal` for data access.
4.  **Remove Redundant Caching Logic**: Remove the caching logic from the `DataLayer` class.
5.  **Test the New Architecture**: Thoroughly test the new architecture to ensure that it is working correctly and that there are no performance regressions.

## 5. Conclusion

The proposed architectural redesign of the data layer will simplify the backtesting engine, improve performance, and provide a clear separation of concerns. By centralizing data access in the `DataPortal`, we can create a more robust and maintainable system.
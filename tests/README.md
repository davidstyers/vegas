# Vegas Backtesting Engine Test Suite

This directory contains a comprehensive test suite for the Vegas backtesting engine. The tests cover all aspects of the engine from database handling to strategy execution and bias prevention.

## Test Organization

The tests are organized into three main categories:

1. **Core Tests**: Test individual components of the backtesting engine
2. **Integration Tests**: Test how components work together in end-to-end scenarios
3. **Specialized Tests**: Test specific aspects like bias prevention and performance

## Test Files

### Core Tests

- `test_database_extended.py`: Tests for database functionality, including initialization, data ingestion, partitioned storage, and queries
- `test_data_layer.py`: Tests for data loading, validation, and querying
- `test_engine.py`: Tests for the backtesting engine, including initialization, execution, and report generation
- `test_portfolio.py`: Tests for portfolio management, position tracking, and performance statistics
- `test_broker.py`: Tests for order execution, slippage models, and commission models

### Integration Tests

- `test_integration.py`: End-to-end tests covering the complete backtesting workflow with different strategies, asset classes, and time frames

### Specialized Tests

- `test_bias_prevention.py`: Tests to ensure the engine prevents common biases like lookahead bias and survivorship bias
- `test_performance.py`: Tests for memory usage, execution speed, and scalability
- `test_robustness.py`: Tests for handling market disruptions, extreme conditions, and error recovery

## Running the Tests

### Running All Tests

To run all tests:

```bash
python run_all_tests.py
```

### Running Specific Test Categories

To run only core tests:

```bash
python run_all_tests.py --type core
```

Available categories:
- `core`: Database, data layer, engine, portfolio, and broker tests
- `integration`: End-to-end integration tests
- `specialized`: Bias prevention, performance, and robustness tests

### Verbose Output

For more detailed output:

```bash
python run_all_tests.py --verbose
```

### Code Coverage

To generate a code coverage report:

```bash
python run_all_tests.py --coverage
```

This will create an HTML coverage report in the `htmlcov` directory.

### Running Individual Test Files

You can also run individual test files directly:

```bash
pytest test_engine.py -v
```

## Test Data Generation

Most tests use synthetic data generated on-the-fly. The data generation functions include:

- Basic OHLCV data generation
- Data with specific patterns for testing strategies
- Data with market disruptions (circuit breakers, trading halts)
- Data with gaps and extreme volatility
- Data with delisted securities

## Requirements

To run the tests, you need:

- pytest
- pytest-cov (for coverage reports)
- numpy
- pandas
- psutil (for memory usage tests)

You can install these dependencies with:

```bash
pip install pytest pytest-cov numpy pandas psutil
``` 
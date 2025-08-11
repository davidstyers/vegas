# Benchmarking the Vegas Engine

This guide explains how to use the benchmarking tool included with Vegas to measure and compare the performance of your backtesting runs.

## Overview

The Vegas benchmarking tool provides a way to:

1. Compare performance between optimized and non-optimized engine configurations
2. Measure the runtime impact of different data sizes and strategy complexities
3. Test engine performance on your specific hardware configuration

## Running the Benchmark

### Basic Usage

To run the benchmark with default settings (optimized mode):

```bash
python benchmark_backtester.py
```

This will run with the default data directory path (`data/us-equities/source`) and use all optimizations.

### Command-Line Options

The benchmark tool supports several command-line options:

#### Data Source Options

```bash
# Use a specific data file instead of a directory
python benchmark_backtester.py --data-file data/sample_data.csv.zst

# Use a specific data directory
python benchmark_backtester.py --data-dir data/custom/path
```

#### Performance Comparison Options

```bash
# Run without optimizations to compare performance
python benchmark_backtester.py --no-optimize

# Run with only a single strategy (instead of multiple)
python benchmark_backtester.py --single-strategy
```

#### Combined Options

```bash
# Compare performance on a specific dataset without optimizations
python benchmark_backtester.py --data-dir data/custom/path --no-optimize
```

## Understanding the Results

The benchmark will output several timing measurements:

1. **Data Loading Time**: Time taken to load data from disk into memory
2. **Backtest Runtime**: Time taken for the backtest execution itself
3. **Total Runtime**: Overall time from start to finish

It also reports key statistics:

- Number of symbols processed
- Total data points processed
- Whether optimizations were enabled
- Performance metrics of each strategy

## Example Output

```
2023-11-15 14:32:18 - vegas.benchmark - INFO - Starting backtester benchmark
2023-11-15 14:32:18 - vegas.benchmark - INFO - Using optimized engine
2023-11-15 14:32:18 - vegas.benchmark - INFO - Loading data from directory: data/us-equities/source
2023-11-15 14:32:23 - vegas.benchmark - INFO - Data loaded in 4.72 seconds
2023-11-15 14:32:23 - vegas.benchmark - INFO - Running backtest from 2022-12-02 to 2023-01-01
2023-11-15 14:32:23 - vegas.benchmark - INFO - Multiple strategies backtested in 2.35 seconds

2023-11-15 14:32:23 - vegas.benchmark - INFO - Results for SimpleMovingAverageStrategy:
2023-11-15 14:32:23 - vegas.benchmark - INFO - Total Return: 3.45%
2023-11-15 14:32:23 - vegas.benchmark - INFO - Sharpe Ratio: 1.28
2023-11-15 14:32:23 - vegas.benchmark - INFO - Number of Trades: 24

2023-11-15 14:32:23 - vegas.benchmark - INFO - Results for MomentumStrategy:
2023-11-15 14:32:23 - vegas.benchmark - INFO - Total Return: 2.17%
2023-11-15 14:32:23 - vegas.benchmark - INFO - Sharpe Ratio: 0.93
2023-11-15 14:32:23 - vegas.benchmark - INFO - Number of Trades: 18

2023-11-15 14:32:23 - vegas.benchmark - INFO - Performance Summary:
2023-11-15 14:32:23 - vegas.benchmark - INFO - Data loading time: 4.72 seconds
2023-11-15 14:32:23 - vegas.benchmark - INFO - Optimizations: Enabled
2023-11-15 14:32:23 - vegas.benchmark - INFO - Symbols: 7
2023-11-15 14:32:23 - vegas.benchmark - INFO - Data points: 25432
```

## Comparing Optimized vs. Non-Optimized

Run both an optimized and non-optimized benchmark to see the performance difference:

```bash
# Run optimized benchmark
python benchmark_backtester.py > optimized_results.txt

# Run non-optimized benchmark
python benchmark_backtester.py --no-optimize > non_optimized_results.txt

# Compare results
diff -y optimized_results.txt non_optimized_results.txt
```

## Tips for Effective Benchmarking

1. **Use realistic data sizes**: Test with datasets similar to what you'll use in production
2. **Test multiple times**: Run benchmarks multiple times and average results for more reliable metrics
3. **Monitor system resources**: Use tools like `top` or Activity Monitor to watch memory usage
4. **Isolate tests**: Close other resource-intensive applications during benchmarking
5. **Test with different hardware**: If possible, benchmark on different machines to understand scalability

## Common Benchmarking Scenarios

### Testing Data Loading Performance

```bash
# Time just the data loading phase with different file counts
python benchmark_backtester.py --single-strategy
```

### Testing Strategy Scaling

```bash
# See how performance scales with multiple strategies
python benchmark_backtester.py
```

### Testing Memory Efficiency

Run with increasingly large datasets while monitoring memory usage to find the practical limits of your system configuration.

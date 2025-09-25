"""Examples demonstrating the Vegas frequency transformation system.

This file shows how to use the new frequency and data transformation capabilities
in the Vegas backtesting engine, including OHLCV resampling and tick bar methods
based on Lopez de Prado's "Advances in Financial Machine Learning".
"""

from datetime import datetime
import polars as pl
from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Signal


class SimpleMovingAverageStrategy(Strategy):
    """Simple moving average crossover strategy for demonstration."""
    
    def initialize(self, context):
        """Initialize strategy parameters."""
        context.fast_window = 10
        context.slow_window = 30
        context.universe = ["AAPL", "MSFT", "GOOGL"]
    
    def handle_data(self, context, data):
        """Generate trading signals based on moving average crossover."""
        signals = []
        
        for symbol in context.universe:
            # Get price history for this symbol
            prices = data.filter(pl.col("symbol") == symbol)["close"]
            
            if len(prices) >= context.slow_window:
                # Calculate moving averages
                fast_ma = prices[-context.fast_window:].mean()
                slow_ma = prices[-context.slow_window:].mean()
                
                # Generate signals
                if fast_ma > slow_ma:
                    signals.append(Signal(symbol=symbol, quantity=100))
                elif fast_ma < slow_ma:
                    signals.append(Signal(symbol=symbol, quantity=-100))
        
        return signals


def run_ohlcv_frequency_examples():
    """Demonstrate OHLCV frequency resampling examples."""
    print("=== OHLCV Frequency Examples ===")
    
    engine = BacktestEngine()
    strategy = SimpleMovingAverageStrategy()
    
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    
    # Example 1: 1-hour bars (default)
    print("\n1. Running backtest with 1-hour bars:")
    results_1h = engine.run(
        start=start,
        end=end, 
        strategy=strategy,
        frequency="1h",
        data_type="ohlcv"
    )
    print(f"Total return: {results_1h.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 2: 4-hour bars
    print("\n2. Running backtest with 4-hour bars:")
    results_4h = engine.run(
        start=start,
        end=end,
        strategy=strategy, 
        frequency="4h",
        data_type="ohlcv"
    )
    print(f"Total return: {results_4h.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 3: Daily bars
    print("\n3. Running backtest with daily bars:")
    results_daily = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="1d",
        data_type="ohlcv"
    )
    print(f"Total return: {results_daily.stats.get('total_return_pct', 0):.2f}%")


def run_tick_bar_examples():
    """Demonstrate tick bar transformation examples."""
    print("\n=== Tick Bar Examples ===")
    
    engine = BacktestEngine()
    strategy = SimpleMovingAverageStrategy()
    
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)  # Shorter period for tick data
    
    # Example 1: Tick bars (1000 ticks per bar)
    print("\n1. Running backtest with tick bars (1000 ticks):")
    results_tick = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="tick:1000",
        data_type="tick"
    )
    print(f"Total return: {results_tick.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 2: Volume bars (10,000 shares per bar)
    print("\n2. Running backtest with volume bars (10K shares):")
    results_volume = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="volume:10000",
        data_type="tick"
    )
    print(f"Total return: {results_volume.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 3: Dollar bars ($100,000 per bar)
    print("\n3. Running backtest with dollar bars ($100K):")
    results_dollar = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="dollar:100000",
        data_type="tick"
    )
    print(f"Total return: {results_dollar.stats.get('total_return_pct', 0):.2f}%")


def run_imbalance_bar_examples():
    """Demonstrate imbalance bar transformation examples."""
    print("\n=== Imbalance Bar Examples ===")
    
    engine = BacktestEngine()
    strategy = SimpleMovingAverageStrategy()
    
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)
    
    # Example 1: Tick imbalance bars
    print("\n1. Running backtest with tick imbalance bars:")
    results_tick_imbalance = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="tick_imbalance:1000",
        data_type="tick"
    )
    print(f"Total return: {results_tick_imbalance.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 2: Volume imbalance bars
    print("\n2. Running backtest with volume imbalance bars:")
    results_volume_imbalance = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="volume_imbalance:10000",
        data_type="tick"
    )
    print(f"Total return: {results_volume_imbalance.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 3: Dollar imbalance bars
    print("\n3. Running backtest with dollar imbalance bars:")
    results_dollar_imbalance = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="dollar_imbalance:100000",
        data_type="tick"
    )
    print(f"Total return: {results_dollar_imbalance.stats.get('total_return_pct', 0):.2f}%")


def run_run_bar_examples():
    """Demonstrate run bar transformation examples."""
    print("\n=== Run Bar Examples ===")
    
    engine = BacktestEngine()
    strategy = SimpleMovingAverageStrategy()
    
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31)
    
    # Example 1: Tick run bars
    print("\n1. Running backtest with tick run bars:")
    results_tick_run = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="tick_run:1000",
        data_type="tick"
    )
    print(f"Total return: {results_tick_run.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 2: Volume run bars
    print("\n2. Running backtest with volume run bars:")
    results_volume_run = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="volume_run:10000",
        data_type="tick"
    )
    print(f"Total return: {results_volume_run.stats.get('total_return_pct', 0):.2f}%")
    
    # Example 3: Dollar run bars
    print("\n3. Running backtest with dollar run bars:")
    results_dollar_run = engine.run(
        start=start,
        end=end,
        strategy=strategy,
        frequency="dollar_run:100000",
        data_type="tick"
    )
    print(f"Total return: {results_dollar_run.stats.get('total_return_pct', 0):.2f}%")


def demonstrate_standalone_transformations():
    """Demonstrate using transformations standalone without backtesting."""
    print("\n=== Standalone Transformation Examples ===")
    
    import polars as pl
    from vegas.data.transform import (
        OHLCVResampler, TickBars, VolumeBars, DollarBars,
        TickImbalanceBars, VolumeImbalanceBars
    )
    
    # Create sample tick data
    sample_tick_data = pl.DataFrame({
        "timestamp": pl.date_range(
            start=datetime(2023, 1, 1, 9, 30),
            end=datetime(2023, 1, 1, 16, 0), 
            interval="1m",
            eager=True
        ),
        "price": [100.0 + i * 0.1 for i in range(391)],  # 391 minutes in trading day
        "volume": [1000 + i * 10 for i in range(391)],
    })
    
    print(f"\nOriginal tick data: {sample_tick_data.height} rows")
    
    # Example 1: Tick bars
    tick_transformer = TickBars(bar_size=100)
    tick_bars = tick_transformer.transform(sample_tick_data)
    print(f"Tick bars (100 ticks): {tick_bars.height} rows")
    
    # Example 2: Volume bars
    volume_transformer = VolumeBars(bar_size=50000)
    volume_bars = volume_transformer.transform(sample_tick_data)
    print(f"Volume bars (50K volume): {volume_bars.height} rows")
    
    # Example 3: Dollar bars
    dollar_transformer = DollarBars(bar_size=500000)
    dollar_bars = dollar_transformer.transform(sample_tick_data)
    print(f"Dollar bars ($500K): {dollar_bars.height} rows")
    
    # Example 4: Tick imbalance bars
    tick_imbalance_transformer = TickImbalanceBars(bar_size=100)
    tick_imbalance_bars = tick_imbalance_transformer.transform(sample_tick_data)
    print(f"Tick imbalance bars: {tick_imbalance_bars.height} rows")


def show_available_frequencies():
    """Show available frequency options for different data types."""
    print("\n=== Available Frequency Options ===")
    
    from vegas.data.transform import FrequencyManager
    from vegas.data.transform.base import DataType
    
    # OHLCV frequencies
    ohlcv_frequencies = FrequencyManager.get_available_frequencies(DataType.OHLCV)
    print("\nOHLCV Data Frequencies:")
    for freq, desc in ohlcv_frequencies.items():
        print(f"  {freq}: {desc}")
    
    # Tick data frequencies
    tick_frequencies = FrequencyManager.get_available_frequencies(DataType.TICK)
    print("\nTick Data Frequencies:")
    for freq, desc in tick_frequencies.items():
        print(f"  {freq}: {desc}")


if __name__ == "__main__":
    """Run all frequency examples."""
    print("Vegas Frequency Transformation Examples")
    print("======================================")
    
    # Show available options
    show_available_frequencies()
    
    # Note: These examples assume you have data loaded in your database
    # In practice, you would need to ingest data first:
    # vegas ingest-ohlcv --directory /path/to/ohlcv/data
    # vegas ingest-tbbo --directory /path/to/tick/data
    
    try:
        # Run OHLCV examples
        run_ohlcv_frequency_examples()
        
        # Run tick bar examples  
        run_tick_bar_examples()
        
        # Run imbalance bar examples
        run_imbalance_bar_examples()
        
        # Run run bar examples
        run_run_bar_examples()
        
        # Demonstrate standalone usage
        demonstrate_standalone_transformations()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have data loaded in the database.")
        print("Use 'vegas ingest-ohlcv' or 'vegas ingest-tbbo' to load data first.")
    
    print("\n=== Examples Complete ===")

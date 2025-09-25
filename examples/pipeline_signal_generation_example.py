#!/usr/bin/env python3
"""
Example demonstrating pipeline usage in signal generation strategies.

This example shows how a strategy can use before_trading_start to:
1. Run a pipeline daily to compute factors and filters
2. Return a dynamic list of symbols for that day's trading universe
3. Generate signals using those symbols in the predict method

The pipeline runs once per day during before_trading_start, and the resulting
symbol universe is used for all timestamps within that day.
"""

from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
import numpy as np

from vegas.engine import BacktestEngine
from vegas.pipeline import Pipeline
from vegas.pipeline.factors import SimpleMovingAverage, Returns
from vegas.pipeline.filters import TopN
from vegas.strategy import Context, Strategy
from vegas.analytics.alpha import Alpha


class PipelineSignalStrategy(Strategy):
    """
    Strategy that uses a pipeline to dynamically select trading universe each day.
    
    This strategy demonstrates the new day-by-day signal generation capability:
    - before_trading_start runs a pipeline to select top momentum stocks
    - Returns the selected symbols as the trading universe for that day
    - predict method generates mean reversion signals for the selected universe
    """
    
    def initialize(self, context: Context):
        """Initialize strategy with pipeline configuration."""
        self.lookback_window = 20
        self.top_n_stocks = 10
        self.mean_reversion_threshold = 0.02
        
        # Create pipeline to select high-momentum stocks
        self.momentum_pipeline = Pipeline(
            columns={
                'returns': Returns(window_length=self.lookback_window),
                'sma_20': SimpleMovingAverage(window_length=20),
                'sma_5': SimpleMovingAverage(window_length=5),
            },
            screen=TopN(
                term=Returns(window_length=self.lookback_window),
                n=self.top_n_stocks,
                ascending=False  # Top momentum stocks
            )
        )
        
        # Attach pipeline to engine (only if not already attached)
        if 'momentum_screen' not in context.engine.attached_pipelines:
            context.engine.attach_pipeline(self.momentum_pipeline, 'momentum_screen')
        
        context.logger = context.engine._logger
        
    def before_trading_start(self, context: Context, data: Dict[str, pl.DataFrame]) -> Optional[List[str]]:
        """
        Run pipeline daily to select trading universe.
        
        This method demonstrates the key new functionality:
        1. Run a pipeline to compute factors and apply filters
        2. Extract the list of symbols that passed the screen
        3. Return this list as the trading universe for the day
        
        Returns:
            List of symbols to trade for this day, or None to use default universe
        """
        try:
            # Run the momentum screening pipeline for today
            pipeline_output = context.engine.pipeline_output('momentum_screen')
            
            if pipeline_output.is_empty():
                context.logger.warning("Pipeline returned no results for today")
                return None
                
            # Extract symbols from pipeline output
            symbols = pipeline_output.get_column('symbol').unique().to_list()
            
            context.logger.info(f"Pipeline selected {len(symbols)} symbols for today: {symbols}")
            
            # Store pipeline results in context for use in predict method
            context.today_pipeline_data = pipeline_output
            
            return symbols
            
        except Exception as e:
            context.logger.error(f"Pipeline execution failed: {e}")
            return None
    
    def predict(self, context: Context, data_portal) -> Dict[str, float]:
        """
        Generate mean reversion signals for the selected universe.
        
        This method uses the data_portal to access historical data for symbols
        selected by today's pipeline and generates signals based on short-term 
        mean reversion logic.
        
        Args:
            context: Strategy context object with portfolio and engine access
            data_portal: DataPortal instance for historical data access
            
        Returns:
            Dictionary mapping symbols to signal values
        """
        signals = {}
        
        # Get universe from today's pipeline results if available
        universe_symbols = None
        if hasattr(context, 'today_pipeline_data') and context.today_pipeline_data is not None:
            try:
                universe_symbols = context.today_pipeline_data.get_column('symbol').unique().to_list()
                context.logger.info(f"Using pipeline universe: {universe_symbols}")
            except Exception as e:
                context.logger.warning(f"Failed to get universe from pipeline: {e}")
        
        # Fallback to a default universe if pipeline fails
        if not universe_symbols:
            try:
                # Get available symbols from data portal as fallback
                universe_symbols = data_portal.get_symbols()[:10]  # Limit to 10 for performance
                context.logger.info(f"Using fallback universe: {universe_symbols}")
            except Exception as e:
                context.logger.warning(f"Failed to get fallback universe: {e}")
                return signals
        
        # Generate signals for each symbol in the universe
        for symbol in universe_symbols:
            try:
                # Get historical data for this symbol
                hist_data = data_portal.history(assets=[symbol], bar_count=20)
                
                if hist_data.is_empty() or hist_data.height < 10:
                    continue
                    
                # Get recent prices
                recent_prices = hist_data.select('close').to_series()
                if recent_prices.len() < 10:
                    continue
                    
                current_price = recent_prices[-1]
                sma_5 = recent_prices.tail(5).mean()
                
                if sma_5 is None or current_price is None:
                    continue
                    
                # Mean reversion signal: sell if price is above SMA, buy if below
                price_deviation = (current_price - sma_5) / sma_5
                
                if abs(price_deviation) > self.mean_reversion_threshold:
                    # Generate contrarian signal: negative when price is high, positive when low
                    signal_strength = -price_deviation
                    signals[symbol] = signal_strength
                    
            except Exception as e:
                # Skip symbol if data processing fails
                context.logger.warning(f"Failed to process {symbol}: {e}")
                continue
                
        return signals


def main():
    """
    Demonstrate pipeline-based signal generation with alpha evaluation.
    
    This example shows the complete workflow:
    1. Strategy defines a pipeline for daily stock selection
    2. generate_signals runs day-by-day, calling before_trading_start each day
    3. Pipeline selects different stocks each day based on momentum
    4. Signals are generated only for the selected stocks
    5. Alpha evaluation to assess predictive power
    """
    print("=== Vegas Pipeline Signal Generation with Alpha Evaluation ===\n")
    
    # Create engine and load some sample data
    engine = BacktestEngine()
    
    # Create strategy
    strategy = PipelineSignalStrategy()
    print(f"1. Strategy: {strategy.__class__.__name__}")
    
    # Initialize strategy to set up parameters (will be done again during signal generation)
    from vegas.strategy import Context
    temp_context = Context()
    temp_context.engine = engine  # Set engine reference for initialization
    strategy.initialize(temp_context)
    
    print(f"   Parameters: lookback={strategy.lookback_window}, top_n={strategy.top_n_stocks}")
    print(f"   Mean reversion threshold: {strategy.mean_reversion_threshold}\n")
    
    # Generate signals for a date range
    # This will now run day-by-day, calling before_trading_start each day
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    print(f"2. Analysis period: {start_date} to {end_date}\n")

    print("3. Generating signals with pipeline-based universe selection...")
    signals_df = engine.generate_signals(
        start=start_date,
        end=end_date,
        strategy=strategy
    )
    
    print(f"   Generated signals for {signals_df.height} timestamps")
    print(f"   Columns: {len(signals_df.columns)} (datetime + {len(signals_df.columns)-1} symbols)")
    print(f"   Sample signals:")
    print(signals_df.head(3))
    print()
    
    # Check if we have any actual signals
    signal_columns = [col for col in signals_df.columns if col != 'datetime']
    if not signal_columns:
        print("   Warning: No signal columns generated!")
        return
    
    # Show daily symbol counts
    daily_counts = (
        signals_df
        .with_columns(pl.col('datetime').dt.date().alias('date'))
        .group_by('date')
        .agg([
            pl.len().alias('timestamps'),
            pl.sum_horizontal(pl.all().exclude(['datetime', 'date']).is_not_null()).alias('active_signals_count')
        ])
        .sort('date')
    )
    print("   Daily active signal counts:")
    print(daily_counts.head(10))
    print()
    
    # Get price data for alpha evaluation
    print("4. Preparing price data for alpha evaluation...")
    
    # Get unique symbols that had signals
    symbols_with_signals = signal_columns
    print(f"   Found {len(symbols_with_signals)} symbols with signals")
    
    # Load data for these symbols
    engine.data_portal.load_data(start_date, end_date, symbols=symbols_with_signals, frequencies=["1h"])
    
    # Get all historical price data at once
    all_price_data = engine.data_portal.history(
        assets=symbols_with_signals, 
        bar_count=10000,  # Large number to get all data
        frequency="1h", 
        end_dt=end_date
    )
    
    if all_price_data.is_empty():
        print("   Warning: No price data available!")
        return
    
    # Pivot the data to get datetime x assets format
    try:
        prices_df = all_price_data.pivot(
            index="timestamp",
            on="symbol", 
            values="close"
        ).rename({"timestamp": "datetime"})
        
        print(f"   Price data shape: {prices_df.shape}")
        print(f"   Sample prices:")
        print(prices_df.head(3))
        print()
    except Exception as e:
        print(f"   Error pivoting price data: {e}")
        return
    
    # Evaluate alpha using the Alpha class
    print("5. Evaluating predictive power (Alpha analysis)...")
    try:
        alpha = Alpha(signals_df, prices_df)
        
        # Calculate forward returns for multiple horizons
        horizons = [1, 5, 20]  # 1 hour, 5 hours, 20 hours ahead
        fwd_returns = alpha.forward_returns(horizons=horizons)
        
        print(f"   Computed forward returns for horizons: {horizons}")
        for h in horizons:
            print(f"   {h}-period forward returns shape: {fwd_returns[h].shape}")
        
        # Evaluate signal quality
        evaluation = alpha.evaluate(horizons=horizons)
        print(f"\n   Alpha evaluation results:")
        print(evaluation)
        print()
        
        # Interpret results
        print("6. Interpretation:")
        for row in evaluation.iter_rows(named=True):
            horizon = row["horizon"]
            ic = row["IC"]
            hit_rate = row["HitRate"]
            
            print(f"   {horizon}-period horizon:")
            print(f"     • Information Coefficient (IC): {ic:.4f}")
            print(f"       {'Strong' if abs(ic) > 0.05 else 'Weak'} {'positive' if ic > 0 else 'negative'} correlation")
            print(f"     • Hit Rate: {hit_rate:.1%}")
            print(f"       {'Above' if hit_rate > 0.5 else 'Below'} random (50%)")
            print()
        
        # Summary
        mean_ic = evaluation["IC"].mean()
        mean_hit_rate = evaluation["HitRate"].mean()
        
        print(f"7. Overall Assessment:")
        print(f"   Average IC across horizons: {mean_ic:.4f}")
        print(f"   Average Hit Rate: {mean_hit_rate:.1%}")
        
        if abs(mean_ic) > 0.02 and mean_hit_rate > 0.52:
            print("   ✅ Pipeline strategy shows promise - consider full backtesting")
        elif abs(mean_ic) > 0.01 or mean_hit_rate > 0.51:
            print("   ⚠️  Pipeline strategy shows weak signal - refinement needed")
        else:
            print("   ❌ Pipeline strategy shows little predictive power - major changes required")
        
        print(f"\n   📊 Pipeline Performance:")
        print(f"   - Dynamic universe selection via momentum pipeline")
        print(f"   - Mean reversion signals on selected universe")
        print(f"   - Evaluated across {len(symbols_with_signals)} symbols and {signals_df.height} timestamps")
        
    except Exception as e:
        print(f"   Error in alpha evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n=== Pipeline Signal Generation Demo Complete ===")  
            


if __name__ == "__main__":
    main()

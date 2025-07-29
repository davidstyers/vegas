"""Example of using the Vegas pipeline system.

This example demonstrates how to use the pipeline system to precompute
data for a momentum-based trading strategy.
"""
import pandas as pd
import numpy as np

from vegas.engine import BacktestEngine
from vegas.strategy import Strategy, Signal
from vegas.pipeline import (
    Pipeline, CustomFactor,
    SimpleMovingAverage, Returns,
    StaticAssets, StandardDeviation
)


class Momentum(CustomFactor):
    """
    Custom factor that calculates momentum based on returns over different periods.
    
    This factor computes a weighted average of 1-month, 3-month, and 6-month returns.
    """
    inputs = ['close']
    window_length = 126  # Approximately 6 months of trading days
    
    def compute(self, today, assets, out, closes):
        # Calculate returns over different periods
        monthly_returns = (closes[-1] / closes[-21] - 1)  # 1-month return
        quarterly_returns = (closes[-1] / closes[-63] - 1)  # 3-month return
        biannual_returns = (closes[-1] / closes[-126] - 1)  # 6-month return
        
        # Weight the returns (more weight to recent periods)
        out[:] = 0.5 * monthly_returns + 0.3 * quarterly_returns + 0.2 * biannual_returns


class MeanReversion(CustomFactor):
    """
    Custom factor that identifies potential mean reversion candidates.
    
    This factor looks for securities that are oversold based on their 
    recent price movements relative to a longer-term moving average.
    """
    inputs = ['close']
    window_length = 30  # 30 days
    
    def compute(self, today, assets, out, closes):
        # Calculate recent price relative to 30-day moving average
        ma30 = np.nanmean(closes, axis=0)
        recent_price = closes[-1]
        
        # Calculate distance from moving average (negative means price is below average)
        distance_from_ma = (recent_price / ma30) - 1
        
        # For mean reversion, we're looking for prices below the moving average
        # so we invert the distance (more negative becomes more positive)
        out[:] = -distance_from_ma


class PipelineStrategy(Strategy):
    """
    A strategy that uses the pipeline to identify trading opportunities.
    
    This strategy combines momentum and mean reversion signals to make trading decisions.
    """
    
    def initialize(self, context):
        """Initialize the strategy with parameters."""
        # Configure the strategy
        context.max_positions = 10
        context.position_size = 0.1  # 10% of portfolio per position
        
        # Create a pipeline
        pipe = make_pipeline()
        
        # Attach the pipeline to the engine
        context._engine.attach_pipeline(pipe, 'my_pipeline')
    
    def before_trading_start(self, context, data):
        """
        Process pipeline results before the trading day starts.
        """
        try:
            # Get the pipeline output
            results = context._engine.pipeline_output('my_pipeline')
            
            if results.empty:
                print("Pipeline returned no results")
                return
            
            # Filter for stocks with high combined score
            context.longs = results[results['combined_score'] > 0.5].index.get_level_values('symbol').tolist()
            context.shorts = results[results['combined_score'] < -0.5].index.get_level_values('symbol').tolist()
            
            # Limit to max positions
            context.longs = context.longs[:context.max_positions]
            context.shorts = context.shorts[:context.max_positions]
            
            print(f"Pipeline identified {len(context.longs)} long and {len(context.shorts)} short candidates")
            
        except Exception as e:
            print(f"Error in before_trading_start: {e}")
    
    def handle_data(self, context, data):
        """Generate trading signals based on pipeline results."""
        signals = []
        
        # Initialize empty lists if they don't exist
        if not hasattr(context, 'longs'):
            context.longs = []
        if not hasattr(context, 'shorts'):
            context.shorts = []
        
        # Get current portfolio positions
        current_positions = set()
        try:
            positions = context.portfolio.get_positions()
            current_positions = {pos.symbol for pos in positions} if positions else set()
        except Exception as e:
            print(f"Error getting positions: {e}")
        
        # Process long signals
        for symbol in context.longs:
            if symbol not in current_positions and symbol in data['symbol'].values:
                # Calculate position size based on portfolio value
                price = data.loc[data['symbol'] == symbol, 'close'].iloc[0]
                quantity = 10  # Default to 10 shares if we can't calculate
                
                if hasattr(context.portfolio, 'current_equity'):
                    position_value = context.portfolio.current_equity * context.position_size
                    quantity = int(position_value / price) if price > 0 else 10
                
                if quantity > 0:
                    signals.append(Signal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity
                    ))
        
        # Process short signals (if allowed)
        for symbol in context.shorts:
            if symbol not in current_positions and symbol in data['symbol'].values:
                # Calculate position size based on portfolio value
                price = data.loc[data['symbol'] == symbol, 'close'].iloc[0]
                quantity = 10  # Default to 10 shares if we can't calculate
                
                if hasattr(context.portfolio, 'current_equity'):
                    position_value = context.portfolio.current_equity * context.position_size
                    quantity = int(position_value / price) if price > 0 else 10
                
                if quantity > 0:
                    signals.append(Signal(
                        symbol=symbol,
                        action="SELL",  # Note: In practice, you would need to check if shorting is enabled
                        quantity=quantity
                    ))
        
        return signals


def make_pipeline():
    """
    Create a pipeline for the strategy.
    
    Returns
    -------
    Pipeline
        The pipeline with all required computations.
    """
    # Create momentum factor
    momentum = Momentum()
    
    # Create mean reversion factor
    mean_reversion = MeanReversion()
    
    # Create volatility factor (standard deviation of returns)
    volatility = StandardDeviation(inputs=['close'], window_length=20)
    
    # Combine the factors into a single score
    # We want high momentum, mean reversion potential, and low volatility
    combined_score = momentum - mean_reversion * 0.5 - volatility * 2
    
    # Create a filter for liquid stocks
    volume_sma = SimpleMovingAverage(
        inputs=['volume'],
        window_length=30
    )
    liquid_stocks = volume_sma > 100000  # Only stocks with decent volume
    
    return Pipeline(
        columns={
            'momentum': momentum,
            'mean_reversion': mean_reversion,
            'volatility': volatility,
            'combined_score': combined_score,
        },
        screen=liquid_stocks
    )


def run_pipeline_backtest():
    """Run a backtest using the pipeline strategy."""
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2020-12-31')
    
    # Create the engine
    engine = BacktestEngine()
    
    # Load data (assuming you have data available)
    #engine.load_data(directory="../db")
    
    # Create and run the strategy
    strategy = PipelineStrategy()
    results = engine.run(start_date, end_date, strategy, initial_capital=100000)
    
    # Print results
    print("Backtest completed")
    print(f"Final portfolio value: ${results['stats']['final_equity']:.2f}")
    print(f"Total return: {results['stats']['total_return_pct']:.2f}%")
    print(f"Number of trades: {results['stats']['num_trades']}")
    
    return results


if __name__ == "__main__":
    run_pipeline_backtest() 
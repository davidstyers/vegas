"""Example of using the Vegas pipeline system.

This example demonstrates how to use the pipeline system to precompute
data for a momentum-based trading strategy.
"""
import polars as pl
from tabulate import tabulate

from vegas.strategy import Strategy, Signal
from vegas.broker.commission import commission
from vegas.pipeline import (
    Pipeline, CustomFactor,
    SimpleMovingAverage, StandardDeviation
)


class PipelineStrategy(Strategy):
    """
    A strategy that uses the pipeline to identify trading opportunities.
    
    This strategy combines momentum and mean reversion signals to make trading decisions.
    """
    
    def initialize(self, context):
        """Initialize the strategy with parameters."""
        # Configure the strategy
        context.max_positions = 10
        context.position_size = 0.01  # 0.1% of portfolio per position
        context.porfolio_postions = []
        
        # Create a pipeline
        pipe = make_pipeline()
        
        # Attach the pipeline to the engine
        context._engine.attach_pipeline(pipe, 'my_pipeline')

        # Set a commission model
        context.set_commission(commission.PerShare(cost_per_share=0.005, min_trade_cost=1.0))
    
    def before_trading_start(self, context, data):
        """
        Process pipeline results before the trading day starts.
        """
        try:
            # Get the pipeline output
            results = context._engine.pipeline_output('my_pipeline')

            longs = (
                results
                .select("symbol")
                .to_series()
                .to_list()
            )

            # Limit to max positions
            context.longs = longs #longs[:context.max_positions]
            
            print(f"Pipeline identified {len(context.longs)} long candidates")
            
        except Exception as e:
            print(f"Error in before_trading_start: {e}")
    
    def handle_data(self, context, data):
        """
        Generate trading signals based on pipeline results.
        """
        signals = []
        
        # Initialize empty lists if they don't exist
        if not hasattr(context, 'longs'):
            context.longs = []
        
        try:
            positions = context.portfolio.get_positions()
            context.porfolio_postions = {pos.symbol for pos in positions} if positions else set()
        except Exception as e:
            print(f"Error getting positions: {e}")
        symbols_in_data = set(data.select("symbol").to_series().to_list())
        
        # Helper to safely get the latest close for a symbol from Polars data
        def get_price(sym: str) -> float | None:
            try:
                s = (
                    data
                    .filter(pl.col("symbol") == sym)
                    .select("close")
                    .to_series()
                )
                if s.len() == 0:
                    return None
                return float(s.item())
            except Exception:
                return None
        
        # Process long signals
        for symbol in context.longs:
            if symbol not in context.porfolio_postions and symbol in symbols_in_data:
                price = get_price(symbol)
                if price is None or price <= 0:
                    continue
                quantity = 10  # Default
                
                if hasattr(context.portfolio, 'current_equity'):
                    position_value = context.portfolio.current_equity * context.position_size
                    quantity = int(position_value / price) if price > 0 else 10
                
                if quantity > 0:
                    signals.append(Signal(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity
                    ))
        
        return signals
     
    def on_market_close(self, context, data, portfolio):
        """
        Process portfolio results after the market closes.
        """
        rows = []
        for pos in portfolio.position_values.values():
            rows.append([
                pos['symbol'],
                f"{pos['buy_price']:.2f}",
                int(pos['qty']),
                f"{pos['current_price']:.2f}",
                f"${pos['current_market_value']:,.2f}",
                f"${pos['pnl']:,.2f}",
                f"{pos['pnl_pct']:.2f}%"
            ])
        print(tabulate(rows, headers=["Symbol", "Buy", "Qty", "Price", "Market Value", "PnL", "PnL %"], tablefmt="fancy_grid"))
        start_portfolio_value = context.portfolio.initial_capital
        current_portfolio_value = context.portfolio.get_portfolio_value()
        pct_change = (current_portfolio_value - start_portfolio_value) / start_portfolio_value
        print(f"Current portfolio value: {context.portfolio.get_portfolio_value():.2f} / {pct_change:.2%}")


class Momentum(CustomFactor):
    """
    Custom factor that calculates momentum based on returns over different periods.
    
    This factor computes a weighted average of 1-month, 3-month, and 6-month returns.
    """
    inputs = ['close']
    window_length = 126  # Approximately 6 months of trading days
    
    def to_expression(self) -> pl.Expr:
        # Calculate returns over different periods
        monthly_returns = pl.col('close').pct_change(n=21).over('symbol')
        quarterly_returns = pl.col('close').pct_change(n=63).over('symbol')
        biannual_returns = pl.col('close').pct_change(n=126).over('symbol')
        
        # Weight the returns (more weight to recent periods)
        return 0.5 * monthly_returns + 0.3 * quarterly_returns + 0.2 * biannual_returns

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
    
    # Create volatility factor (standard deviation of returns)
    volatility = StandardDeviation(inputs=['close'], window_length=20)
    
    # Create a filter for liquid stocks
    volume_sma = SimpleMovingAverage(
        inputs=['volume'],
        window_length=30
    )

    price_sma = SimpleMovingAverage(
        inputs=['close'],
        window_length=10
    )

    liquid_stocks = volume_sma > 100_000  # Only stocks with decent volume
    no_pennies = price_sma > 2.00
    
    return Pipeline(
        columns={
            'momentum': momentum,
            'volatility': volatility
        },
        screen=liquid_stocks & no_pennies & momentum.top(10)
    )

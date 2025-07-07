"""Vegas Wave Strategy.

Vegas Wave trading system.

"""


from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta

from vegas.strategy import Strategy, Signal, Context


class VegasWave(Strategy):
    
    def initialize(self, context: Context) -> None:
        """Initialize the strategy with parameters.
        
        Args:
            context: Strategy context
        """
        # Strategy parameters
        context.symbols = None
        
        # Track the last trading day to avoid duplicate orders
        context.last_trading_day = None
    
    def before_trading_start(self, context: Context, data: Dict[str, pd.DataFrame]) -> None:
        """Execute pre-trading logic.
        
        Args:
            context: Strategy context
            data: Market data by symbol
        """
        # Nothing to do here for this strategy
        pass
    
    def handle_data(self, context: Context, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Process market data and generate trading signals.
        
        Args:
            context: Strategy context
            data: Market data by symbol
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if our symbol is in the data
        if context.symbol not in data:
            return signals
        
        # Get the latest timestamp from the data
        symbol_data = data[context.symbol]
        if symbol_data.empty:
            return signals
            
        latest_timestamp = symbol_data['timestamp'].iloc[-1]
        current_date = latest_timestamp.date()
        
        # Skip if we've already processed this trading day
        if context.last_trading_day == current_date:
            return signals
            
        # Update the last trading day
        context.last_trading_day = current_date
        
        # Get current position
        current_position = 0
        portfolio = context.portfolio
        if portfolio is not None:
            # added the 1 hr timedelta to get the previous position
            position = portfolio.positions_history.get(latest_timestamp - timedelta(hours=1), {}).get(context.symbol, {})
            if position:
                current_position = position.get('quantity', 0)
        
        # Check if today is Monday (weekday 0) - BUY
        if latest_timestamp.weekday() == 0:  # Monday
            signals.append(Signal(
                symbol=context.symbol,
                action='buy',
                quantity=context.shares_to_buy
            ))
        
        # Check if today is Friday (weekday 4) - SELL
        elif latest_timestamp.weekday() == 4:  # Friday
            # Only sell if we have a position
            if current_position != 0:
                signals.append(Signal(
                    symbol=context.symbol,
                    action='sell',
                    quantity=current_position  # Sell all shares
                ))
        
        return signals

def run_backtest():
    """Run a backtest with the Weekly Apple Trader strategy."""
    from vegas.engine import BacktestEngine
    from datetime import datetime, timedelta
    
    # Create backtest engine
    engine = BacktestEngine()
    
    # Load sample data
    engine.load_data("data/sample_data.csv.zst")
    
    # Define strategy
    strategy = VegasWave()
    
    # Set date range (adjust based on your data)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 1, 31)
    
    # Run backtest
    results = engine.run(
        start=start_date,
        end=end_date,
        strategy=strategy,
        initial_capital=50000.0,
        benchmark_symbol='SPY'  # Benchmark against S&P 500
    )
    
    # Analyze and visualize results
    results.plot_equity_curve()
    results.plot_drawdown()
    
    # Export results
    results.export_results('json', 'weekly_aapl_results.json')
    
    print(f"Backtest completed with total return: {results.stats['total_return_pct']:.2f}%")
    print(f"Total Return: {results.stats['total_return']:.2f}")
    print(f"Sharpe Ratio: {results.stats['sharpe_ratio']:.2f}")
    
    return results


if __name__ == '__main__':
    run_backtest() 
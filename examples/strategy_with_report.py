#!/usr/bin/env python3
"""Example strategy that demonstrates how to generate QuantStats reports.

This strategy shows how to use the create_tearsheet function
from the analytics module to create performance reports directly from
strategy code.
"""

from datetime import datetime
from vegas.strategy import Strategy, Signal


class StrategyWithReport(Strategy):
    """Simple moving average strategy that generates its own performance report."""
    
    def initialize(self, context):
        """Initialize the strategy with parameters."""
        context.sma_short = 10
        context.sma_long = 30
        context.symbol = "AAPL"
        
    def handle_data(self, context, data):
        """Generate trading signals based on moving average crossover."""
        signals = []
        
        if context.symbol in data.index:
            symbol_data = data.loc[context.symbol]
            
            # Calculate moving averages
            if len(symbol_data) >= context.sma_long:
                sma_short = symbol_data['close'].rolling(context.sma_short).mean().iloc[-1]
                sma_long = symbol_data['close'].rolling(context.sma_long).mean().iloc[-1]
                
                # Generate signals based on crossover
                if sma_short > sma_long:
                    # Buy signal
                    signals.append(Signal(
                        symbol=context.symbol,
                        quantity=10,
                        order_type='market'
                    ))
                elif sma_short < sma_long:
                    # Sell signal
                    signals.append(Signal(
                        symbol=context.symbol,
                        quantity=-10,
                        order_type='market'
                    ))
        
        return signals
    
    def analyze(self, context, results):
        """Generate a QuantStats report after the backtest completes.
        
        This method is called automatically by the engine after the backtest
        finishes, allowing strategies to perform their own analysis and
        generate reports.
        """
        # Generate a QuantStats report using the Results object's create_tearsheet method
        try:
            results.create_tearsheet(
                title="StrategyWithReport Performance Report",
                benchmark_symbol="SPY",
                output_file="reports/strategy_with_report.html",
                output_format="html"
            )
            print("✅ QuantStats report generated successfully!")
            print("📊 Report saved to: reports/strategy_with_report.html")
        except Exception as e:
            print(f"❌ Failed to generate QuantStats report: {e}")


# Example usage in a script
if __name__ == "__main__":
    from vegas.engine import BacktestEngine
    from datetime import datetime
    
    # Create and run backtest
    engine = BacktestEngine()
    
    # Load some sample data (you would replace this with your actual data)
    # engine.load_data("path/to/your/data.csv")
    
    # Run the backtest
    results = engine.run(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        strategy=StrategyWithReport(),
        initial_capital=100000
    )
    
    # The analyze method will be called automatically, but you can also
    # generate additional reports manually if needed
    results.create_tearsheet(
        title="Manual Report",
        benchmark_symbol="QQQ",
        output_file="reports/manual_report.html",
        output_format="html"
    )

"""
Example demonstrating side field usage in tick bars.

This example shows how to use the new side field functionality to create
bars with buy/sell aggressor tracking.
"""

import polars as pl
from datetime import datetime, timedelta
from vegas.data.transform import TickBars, VolumeBars, DollarBars, TickImbalanceBars

def create_sample_tick_data():
    """Create sample tick data with side field."""
    # Generate sample tick data
    timestamps = []
    prices = []
    sizes = []
    sides = []
    symbols = []
    
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    base_price = 100.0
    
    for i in range(1000):
        timestamps.append(base_time + timedelta(seconds=i))
        # Simulate price movement
        price_change = (i % 10 - 5) * 0.01  # Small price changes
        prices.append(base_price + price_change)
        base_price += price_change
        
        # Random size between 100 and 1000
        sizes.append(100 + (i % 900))
        
        # Simulate side field: Bid (buy aggressor), Ask (sell aggressor), or None
        side_choice = i % 3
        if side_choice == 0:
            sides.append("Bid")  # Buy aggressor
        elif side_choice == 1:
            sides.append("Ask")  # Sell aggressor
        else:
            sides.append(None)  # Unknown side
            
        symbols.append("AAPL")
    
    return pl.DataFrame({
        "timestamp": timestamps,
        "symbol": symbols,
        "price": prices,
        "size": sizes,
        "side": sides
    })

def demonstrate_side_based_bars():
    """Demonstrate the new side-based bar functionality."""
    print("Creating sample tick data with side field...")
    tick_data = create_sample_tick_data()
    print(f"Sample data shape: {tick_data.shape}")
    print(f"Sample data columns: {tick_data.columns}")
    print(f"Side field values: {tick_data['side'].unique()}")
    print()
    
    # Demonstrate TickBars with side field
    print("=== TickBars (with side field) ===")
    tick_bars = TickBars(bar_size=100)
    tick_bars_result = tick_bars.transform(tick_data)
    print(f"Tick bars result shape: {tick_bars_result.shape}")
    print(f"Tick bars columns: {tick_bars_result.columns}")
    print("Sample tick bars:")
    print(tick_bars_result.head())
    print()
    
    # Demonstrate VolumeBars with side field
    print("=== VolumeBars (with side field) ===")
    volume_bars = VolumeBars(bar_size=5000)
    volume_bars_result = volume_bars.transform(tick_data)
    print(f"Volume bars result shape: {volume_bars_result.shape}")
    print(f"Volume bars columns: {volume_bars_result.columns}")
    print("Sample volume bars:")
    print(volume_bars_result.head())
    print()
    
    # Demonstrate DollarBars with side field
    print("=== DollarBars (with side field) ===")
    dollar_bars = DollarBars(bar_size=50000)
    dollar_bars_result = dollar_bars.transform(tick_data)
    print(f"Dollar bars result shape: {dollar_bars_result.shape}")
    print(f"Dollar bars columns: {dollar_bars_result.columns}")
    print("Sample dollar bars:")
    print(dollar_bars_result.head())
    print()
    
    # Demonstrate TickImbalanceBars with side field
    print("=== TickImbalanceBars (with side field) ===")
    imbalance_bars = TickImbalanceBars(bar_size=50, expected_imbalance=0.0)
    imbalance_bars_result = imbalance_bars.transform(tick_data)
    print(f"Imbalance bars result shape: {imbalance_bars_result.shape}")
    print(f"Imbalance bars columns: {imbalance_bars_result.columns}")
    print("Sample imbalance bars:")
    print(imbalance_bars_result.head())
    print()
    
    # Show aggressor statistics
    print("=== Aggressor Statistics ===")
    if not tick_bars_result.is_empty():
        total_buy = tick_bars_result["buy_aggressor"].sum()
        total_sell = tick_bars_result["sell_aggressor"].sum()
        total_unknown = tick_bars_result["unknown_side"].sum()
        total_ticks = total_buy + total_sell + total_unknown
        
        print(f"Total buy aggressors: {total_buy}")
        print(f"Total sell aggressors: {total_sell}")
        print(f"Total unknown sides: {total_unknown}")
        print(f"Buy aggressor ratio: {total_buy / total_ticks:.2%}")
        print(f"Sell aggressor ratio: {total_sell / total_ticks:.2%}")
        print(f"Unknown side ratio: {total_unknown / total_ticks:.2%}")

if __name__ == "__main__":
    demonstrate_side_based_bars()

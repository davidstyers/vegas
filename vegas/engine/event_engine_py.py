"""Pure Python implementation of the event-driven backtesting engine.

This module provides a fallback implementation for environments where
Cython is not available or compilation fails.
"""

import pandas as pd
import numpy as np
import logging

# Define event types
EVENT_BEFORE_TRADING = 1
EVENT_MARKET_OPEN = 2
EVENT_MARKET_CLOSE = 3
EVENT_TICK = 4
EVENT_BAR = 5
EVENT_TRADE = 6


class EventDrivenEngine:
    """Pure Python implementation of the event-driven backtesting engine."""
    
    def __init__(self, strategy, portfolio, data_layer, logger=None, debug=False):
        """Initialize the event-driven engine with strategy, portfolio and data.
        
        Args:
            strategy: Strategy object with event callbacks
            portfolio: Portfolio object
            data_layer: DataLayer object
            logger: Logger object
            debug: Enable debug logging
        """
        self.strategy = strategy
        self.portfolio = portfolio
        self.data_layer = data_layer
        self.logger = logger or logging.getLogger('vegas.engine.event')
        self.events_by_timestamp = {}
        self.debug = debug
    
    def run_event_driven_backtest(self, events_df):
        """Run the event-driven backtest using a DataFrame of events.
        
        Args:
            events_df: DataFrame with events, containing columns:
                      - timestamp: datetime object
                      - type: integer event type
                      - data: optional data object

        Returns:
            Dictionary with backtest results
        """
        # Set up tracking variables
        order_book = {}
        results = {}
        
        # Sort events chronologically
        events_df = events_df.sort_values('timestamp')
        n_events = len(events_df)
        
        if self.debug:
            self.logger.debug(f"Running event-driven backtest with {n_events} events")
        
        # Main event loop
        for _, event in events_df.iterrows():
            timestamp = event['timestamp']
            event_type = event['type']
            
            # Get the relevant data slice for this timestamp
            if 'data' in event and event['data'] is not None:
                data_slice = event['data']
            else:
                data_slice = self.data_layer.get_data_for_timestamp(timestamp)
            
            # Process different event types
            if event_type == EVENT_BEFORE_TRADING:
                if hasattr(self.strategy, 'before_trading_start'):
                    if self.debug:
                        self.logger.debug(f"Calling before_trading_start at {timestamp}")
                    self.strategy.before_trading_start(self.strategy.context, data_slice)
            
            elif event_type == EVENT_MARKET_OPEN:
                if hasattr(self.strategy, 'on_market_open'):
                    if self.debug:
                        self.logger.debug(f"Calling on_market_open at {timestamp}")
                    self.strategy.on_market_open(self.strategy.context, data_slice, self.portfolio)
            
            elif event_type == EVENT_MARKET_CLOSE:
                if hasattr(self.strategy, 'on_market_close'):
                    if self.debug:
                        self.logger.debug(f"Calling on_market_close at {timestamp}")
                    self.strategy.on_market_close(self.strategy.context, data_slice, self.portfolio)
            
            elif event_type == EVENT_BAR:
                if hasattr(self.strategy, 'on_bar'):
                    if self.debug:
                        self.logger.debug(f"Calling on_bar at {timestamp}")
                    self.strategy.on_bar(self.strategy.context, data_slice)
            
            elif event_type == EVENT_TICK:
                if hasattr(self.strategy, 'on_tick'):
                    if self.debug:
                        self.logger.debug(f"Calling on_tick at {timestamp}")
                    self.strategy.on_tick(self.strategy.context, data_slice)
            
            # Always call handle_data for this timestamp
            if hasattr(self.strategy, 'handle_data'):
                if self.debug:
                    self.logger.debug(f"Calling handle_data at {timestamp}")
                signals = self.strategy.handle_data(self.strategy.context, data_slice)
                
                if signals:
                    # Convert signals to DataFrame
                    signal_records = []
                    for signal in signals:
                        signal_records.append({
                            'timestamp': timestamp,
                            'symbol': signal.symbol,
                            'action': signal.action,
                            'quantity': signal.quantity,
                            'price': signal.price
                        })
                    
                    if signal_records:
                        signal_df = pd.DataFrame(signal_records)
                        # Process signals and create transactions
                        transactions = self._create_transactions_from_signals(signal_df, data_slice)
                        
                        if not transactions.empty:
                            # Update portfolio with transactions
                            self.portfolio.update_from_transactions(timestamp, transactions, data_slice)
                            
                            if self.debug:
                                self.logger.debug(f"Executed {len(transactions)} transactions at {timestamp}")
            
            # Update portfolio state at every event, even without transactions
            if not data_slice.empty:
                self.portfolio.update_from_transactions(timestamp, pd.DataFrame(), data_slice)
        
        if self.debug:
            self.logger.debug("Event-driven backtest completed")
        
        return {
            'stats': self.portfolio.get_stats(),
            'equity_curve': self.portfolio.get_equity_curve(),
            'transactions': self.portfolio.get_transactions(),
            'positions': self.portfolio.get_positions(),
            'success': True
        }
    
    def _create_transactions_from_signals(self, signals, market_data):
        """Convert signals to transactions.
        
        Args:
            signals: DataFrame with signals
            market_data: DataFrame with current market data
            
        Returns:
            DataFrame with transactions
        """
        transactions = []
        
        # Create a lookup for current prices
        price_lookup = {}
        if not market_data.empty and 'symbol' in market_data.columns and 'close' in market_data.columns:
            price_lookup = market_data.set_index('symbol')['close'].to_dict()
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            
            # Skip if symbol not in current market data
            if symbol not in price_lookup:
                continue
                
            # Get execution price
            price = signal.get('price') if signal.get('price') is not None else price_lookup[symbol]
            
            # Convert action to quantity (positive for buy, negative for sell)
            if action.lower() == 'sell':
                quantity = -abs(quantity)
                
            # Add a simplified commission
            commission = abs(quantity * price * 0.001)  # 0.1% commission
            
            transactions.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission
            })
        
        return pd.DataFrame(transactions) if transactions else pd.DataFrame()


def generate_events(start_date, end_date, data_layer, event_types=None, debug=False):
    """Generate events for the event-driven backtest.
    
    Args:
        start_date: Start date for the backtest
        end_date: End date for the backtest
        data_layer: DataLayer object
        event_types: List of event types to generate
        debug: Enable debug logging
        
    Returns:
        DataFrame with events
    """
    if event_types is None:
        event_types = ['before_trading', 'market_open', 'market_close']
        
    events = []
    
    # Get unique trading days in the data
    trading_days = data_layer.get_trading_days(start_date, end_date)
    
    for date in trading_days:
        # Generate before trading start events (9:00 AM)
        if 'before_trading' in event_types:
            events.append({
                'timestamp': pd.Timestamp(date).replace(hour=9, minute=0),
                'type': EVENT_BEFORE_TRADING,
                'data': None
            })
        
        # Generate market open events (9:30 AM)
        if 'market_open' in event_types:
            events.append({
                'timestamp': pd.Timestamp(date).replace(hour=9, minute=30),
                'type': EVENT_MARKET_OPEN,
                'data': None
            })
            
        # Generate bar events throughout the day if needed
        if 'bar' in event_types:
            for hour in range(9, 16):
                for minute in range(0, 60, 30):  # 30-minute bars
                    # Skip pre-market hours
                    if hour == 9 and minute < 30:
                        continue
                    
                    # Skip after-hours
                    if hour == 16 and minute > 0:
                        continue
                        
                    events.append({
                        'timestamp': pd.Timestamp(date).replace(hour=hour, minute=minute),
                        'type': EVENT_BAR,
                        'data': None
                    })
        
        # Generate market close events (4:00 PM)
        if 'market_close' in event_types:
            events.append({
                'timestamp': pd.Timestamp(date).replace(hour=16, minute=0),
                'type': EVENT_MARKET_CLOSE,
                'data': None
            })
    
    if not events:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['timestamp', 'type', 'data'])
    
    # Convert to DataFrame and sort by timestamp
    events_df = pd.DataFrame(events).sort_values('timestamp')
    
    if debug:
        print(f"Generated {len(events_df)} events from {start_date} to {end_date}")
    
    return events_df 
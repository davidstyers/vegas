"""Tests for the Vegas pipeline system."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from vegas.data import DataLayer
from vegas.pipeline import (
    Pipeline, PipelineEngine, CustomFactor,
    SimpleMovingAverage, Returns, StaticAssets
)


class TestPipeline(unittest.TestCase):
    """Test the core functionality of the pipeline system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a DataLayer with test data
        self.data_layer = DataLayer(data_dir="tests/db")
        
        # Create test data
        self.create_test_data()
        
        # Create a PipelineEngine
        self.engine = PipelineEngine(self.data_layer)
    
    def create_test_data(self):
        """Create test data for the pipeline tests."""
        # Create a simple DataFrame with test data
        dates = pd.date_range(start='2020-01-01', end='2020-01-10')
        symbols = ['AAPL', 'MSFT', 'GOOG']
        
        data = []
        for date in dates:
            for symbol in symbols:
                # Generate some test prices that trend upward
                base_price = 100.0
                if symbol == 'AAPL':
                    price = base_price + (date.dayofyear - dates[0].dayofyear) * 2.0
                elif symbol == 'MSFT':
                    price = base_price + (date.dayofyear - dates[0].dayofyear) * 1.5
                else:  # GOOG
                    price = base_price + (date.dayofyear - dates[0].dayofyear) * 1.0
                
                # Add some random noise
                price = price * (1.0 + np.random.normal(0, 0.02))
                
                # Add volume data
                volume = int(np.random.randint(10000, 1000000))
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': volume
                })
        
        # Convert to DataFrame
        self.test_data = pd.DataFrame(data)
        
        # Set the test data in the data layer
        self.data_layer._data = self.test_data
        
        # Also need to explicitly initialize the data_layer's internal structures
        # since we've manually set the _data attribute
        self.data_layer._initialized = True
        
        # Mock the get_trading_days method to return our test dates
        def mock_get_trading_days(start, end):
            return dates[(dates >= start) & (dates <= end)]
        
        self.data_layer.get_trading_days = mock_get_trading_days
        
        # Mock the get_universe method to return our test symbols
        def mock_get_universe(date):
            return symbols
        
        self.data_layer.get_universe = mock_get_universe
        
        # Mock the get_data_for_backtest method to return our test data
        def mock_get_data_for_backtest(start, end):
            return self.test_data[(self.test_data['timestamp'] >= start) & 
                                  (self.test_data['timestamp'] <= end)].copy()
        
        self.data_layer.get_data_for_backtest = mock_get_data_for_backtest
    
    def test_simple_pipeline(self):
        """Test creating and running a simple pipeline."""
        # Create a simple pipeline
        pipeline = Pipeline(
            columns={
                'returns': Returns(inputs=['close'], window_length=2),
                'sma': SimpleMovingAverage(inputs=['close'], window_length=3)
            }
        )
        
        # Run the pipeline
        start_date = pd.Timestamp('2020-01-05')
        end_date = pd.Timestamp('2020-01-10')
        results = self.engine.run_pipeline(pipeline, start_date, end_date)
        
        # Check that we got some results
        self.assertFalse(results.empty, "Pipeline should return non-empty results")
        
        # If we have a MultiIndex, check that we have the expected levels
        if isinstance(results.index, pd.MultiIndex):
            self.assertIn('date', results.index.names)
            self.assertIn('symbol', results.index.names)
            
            # Check that we have the expected columns
            self.assertIn('returns', results.columns)
            self.assertIn('sma', results.columns)
    
    def test_custom_factor(self):
        """Test creating and using a custom factor."""
        
        class TestFactor(CustomFactor):
            """A test factor that computes the ratio of close to open price."""
            inputs = ['close', 'open']
            window_length = 1
            
            def compute(self, today, assets, out, closes, opens):
                out[:] = closes[-1] / opens[-1]
        
        # Create a pipeline with the custom factor
        pipeline = Pipeline(
            columns={
                'close_to_open': TestFactor()
            }
        )
        
        # Run the pipeline
        start_date = pd.Timestamp('2020-01-05')
        end_date = pd.Timestamp('2020-01-10')
        results = self.engine.run_pipeline(pipeline, start_date, end_date)
        
        # Check that we got some results
        self.assertFalse(results.empty, "Pipeline should return non-empty results")
        
        # Check that we have the expected column
        self.assertIn('close_to_open', results.columns)
        
        # Check that the values make sense (close should be ~1% higher than open on average)
        self.assertGreater(results['close_to_open'].mean(), 0.98)
        self.assertLess(results['close_to_open'].mean(), 1.02)
    
    def test_filter(self):
        """Test using a filter in a pipeline."""
        # Create a filter for AAPL only
        aapl_filter = StaticAssets(['AAPL'])
        
        # Create a pipeline with a screen
        pipeline = Pipeline(
            columns={
                'returns': Returns(inputs=['close'], window_length=2)
            },
            screen=aapl_filter
        )
        
        # Run the pipeline
        start_date = pd.Timestamp('2020-01-05')
        end_date = pd.Timestamp('2020-01-10')
        results = self.engine.run_pipeline(pipeline, start_date, end_date)
        
        # Check that we got some results
        self.assertFalse(results.empty, "Pipeline should return non-empty results")
        
        # Check that we only have AAPL in the results
        if isinstance(results.index, pd.MultiIndex):
            symbols = results.index.get_level_values('symbol').unique()
            self.assertEqual(len(symbols), 1)
            self.assertEqual(symbols[0], 'AAPL')


if __name__ == '__main__':
    unittest.main() 
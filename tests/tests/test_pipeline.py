import pytest
import polars as pl
from polars.testing import assert_frame_equal
from datetime import datetime, date
from vegas.pipeline.engine import PipelineEngine
from vegas.pipeline.pipeline import Pipeline
from vegas.pipeline.factors.basic import SimpleMovingAverage, Returns
from vegas.pipeline.filters.basic import StaticAssets
from vegas.data.data_layer import DataLayer

@pytest.fixture
def mock_data_layer():
    """
    Create a mock data layer with sample market data.
    """
    class MockDataLayer(DataLayer):
        def get_data_for_backtest(self, start_date, end_date):
            return pl.DataFrame({
                'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
                'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
                'close': [100.0, 101.0, 102.0, 103.0, 104.0],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

    return MockDataLayer()

def test_run_pipeline_with_factors(mock_data_layer):
    """
    Test running a pipeline with a simple factor.
    """
    engine = PipelineEngine(mock_data_layer)
    
    # Create a pipeline with a 3-day SMA factor
    pipeline = Pipeline(columns={
        'sma_3': SimpleMovingAverage(window_length=3)
    })
    
    # Run the pipeline
    result = engine.run_pipeline(pipeline, date(2023, 1, 3), date(2023, 1, 5))
    
    # Check the results
    expected = pl.DataFrame({
        'date': [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'sma_3': [101.0, 102.0, 103.0]
    })
    
    assert_frame_equal(result, expected)

def test_run_pipeline_with_screen(mock_data_layer):
    """
    Test running a pipeline with a screen.
    """
    engine = PipelineEngine(mock_data_layer)
    
    # Create a pipeline with a 3-day SMA factor and a static asset screen
    pipeline = Pipeline(
        columns={'sma_3': SimpleMovingAverage(window_length=3)},
        screen=StaticAssets(['AAPL'])
    )
    
    # Run the pipeline
    result = engine.run_pipeline(pipeline, date(2023, 1, 3), date(2023, 1, 5))
    
    # Check the results
    expected = pl.DataFrame({
        'date': [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
        'symbol': ['AAPL', 'AAPL', 'AAPL'],
        'sma_3': [101.0, 102.0, 103.0]
    })
    
    assert_frame_equal(result, expected)

def test_run_pipeline_with_returns(mock_data_layer):
    """
    Test running a pipeline with a returns factor.
    """
    engine = PipelineEngine(mock_data_layer)
    
    # Create a pipeline with a 2-day returns factor
    pipeline = Pipeline(columns={
        'returns': Returns(window_length=2)
    })
    
    # Run the pipeline
    result = engine.run_pipeline(pipeline, date(2023, 1, 2), date(2023, 1, 5))
    
    # Check the results
    expected = pl.DataFrame({
        'date': [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)],
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'returns': [0.01, 0.0099, 0.0098, 0.0097]
    })
    
    assert_frame_equal(result, expected, atol=1e-4)
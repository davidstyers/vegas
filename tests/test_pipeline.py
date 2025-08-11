from datetime import date, datetime

import polars as pl
import pytest

from vegas.pipeline.engine import PipelineEngine
from vegas.pipeline.factors.basic import Returns, SimpleMovingAverage
from vegas.pipeline.filters.basic import StaticAssets
from vegas.pipeline.pipeline import Pipeline


@pytest.fixture
def mock_data_portal():
    """Mock DataPortal that returns deterministic daily bars and ignores end_dt."""

    class MockDataPortal:
        def __init__(self):
            self.timezone = "US/Eastern"

        def history(
            self, assets=None, fields=None, bar_count=1, frequency="1d", end_dt=None
        ):
            # Build a 5-day sample series 2023-01-01..2023-01-05 for AAPL
            ts = [datetime(2023, 1, d) for d in range(1, 6)]
            df = pl.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": ["AAPL"] * 5,
                    "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "volume": [1000, 1100, 1200, 1300, 1400],
                }
            )
            # Return the last bar_count rows
            return df.tail(bar_count)

    return MockDataPortal()


def test_run_pipeline_with_factors(mock_data_portal):
    engine = PipelineEngine(mock_data_portal)

    pipeline = Pipeline(
        columns={"sma_3": SimpleMovingAverage(window_length=3)}, frequency="1d"
    )

    result = engine.run_pipeline(pipeline, date(2023, 1, 5), date(2023, 1, 5))

    # Expect last 3 days present; last row SMA should be (102+103+104)/3 = 103.0
    assert result.height == 3
    assert result.columns[0] == "timestamp"
    assert result.columns[1] == "symbol"
    assert result.columns[2] == "sma_3"
    assert abs(float(result.item(-1, "sma_3")) - 103.0) < 1e-6


def test_run_pipeline_with_screen(mock_data_portal):
    engine = PipelineEngine(mock_data_portal)

    pipeline = Pipeline(
        columns={"sma_3": SimpleMovingAverage(window_length=3)},
        screen=StaticAssets(["AAPL"]),
        frequency="1d",
    )

    result = engine.run_pipeline(pipeline, date(2023, 1, 5), date(2023, 1, 5))

    # Same expectations as above; screen retains AAPL
    assert result.height == 3
    assert "sma_3" in result.columns
    assert abs(float(result.item(-1, "sma_3")) - 103.0) < 1e-6


def test_run_pipeline_with_returns(mock_data_portal):
    engine = PipelineEngine(mock_data_portal)

    pipeline = Pipeline(columns={"returns": Returns(window_length=2)}, frequency="1d")

    result = engine.run_pipeline(pipeline, date(2023, 1, 5), date(2023, 1, 5))

    # Last 2 days present; last row return = 104/103 - 1
    assert result.height == 2
    expected_ret = 104.0 / 103.0 - 1.0
    assert abs(float(result.item(-1, "returns")) - expected_ret) < 1e-9

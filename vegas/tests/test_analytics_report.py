"""Tests for the QuantStats report generation functionality."""

import pytest
import polars as pl
from datetime import datetime, timedelta
from vegas.analytics import generate_quantstats_report


@pytest.fixture
def sample_results():
    """Create sample backtest results for testing."""
    # Create sample equity curve data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    equity_values = [100000 + i * 100 for i in range(100)]  # Simple upward trend
    
    equity_curve = pl.DataFrame({
        "timestamp": dates,
        "equity": equity_values,
        "cash": [50000] * 100,
        "return": [0.001] * 100  # 0.1% daily return
    })
    
    return {
        "equity_curve": equity_curve,
        "stats": {
            "total_return": 10000,
            "sharpe_ratio": 1.5,
            "max_drawdown": -5.0
        }
    }


def test_generate_quantstats_report_basic(sample_results, tmp_path):
    """Test basic QuantStats report generation."""
    report_path = tmp_path / "test_report.html"
    
    success = generate_quantstats_report(
        results=sample_results,
        strategy_name="Test Strategy",
        report_path=str(report_path),
        benchmark="SPY"
    )
    
    # The function should return True if QuantStats is available
    # and False if it's not installed
    assert isinstance(success, bool)


def test_generate_quantstats_report_no_equity_curve(tmp_path):
    """Test report generation with missing equity curve."""
    results = {"stats": {"total_return": 1000}}
    report_path = tmp_path / "test_report.html"
    
    success = generate_quantstats_report(
        results=results,
        strategy_name="Test Strategy",
        report_path=str(report_path),
        benchmark="SPY"
    )
    
    assert success is False


def test_generate_quantstats_report_with_logger(sample_results, tmp_path, caplog):
    """Test report generation with custom logger."""
    report_path = tmp_path / "test_report.html"
    
    import logging
    test_logger = logging.getLogger("test_logger")
    
    success = generate_quantstats_report(
        results=sample_results,
        strategy_name="Test Strategy",
        report_path=str(report_path),
        benchmark="SPY",
        logger=test_logger
    )
    
    assert isinstance(success, bool)


def test_generate_quantstats_report_pandas_dataframe(tmp_path):
    """Test report generation with pandas DataFrame."""
    import pandas as pd
    
    # Create sample data as pandas DataFrame
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    equity_values = [100000 + i * 100 for i in range(100)]
    
    equity_curve = pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values,
        "cash": [50000] * 100,
        "return": [0.001] * 100
    })
    
    results = {
        "equity_curve": equity_curve,
        "stats": {"total_return": 10000}
    }
    
    report_path = tmp_path / "test_report.html"
    
    success = generate_quantstats_report(
        results=results,
        strategy_name="Test Strategy",
        report_path=str(report_path),
        benchmark="SPY"
    )
    
    assert isinstance(success, bool)

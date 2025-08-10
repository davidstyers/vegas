"""Pipeline engine implementation for the Vegas backtesting system.

This module defines the PipelineEngine class that computes pipelines.
"""
import pytz
from typing import Dict, List, Optional, Union, Any, Tuple
import polars as pl
from datetime import datetime, timedelta, date
import logging

from vegas.pipeline.pipeline import Pipeline
from vegas.pipeline.terms import Term, Factor, Filter, Classifier


class PipelineEngine:
    """
    An engine for computing Pipelines.
    
    The PipelineEngine is responsible for executing pipeline computations
    and returning the results as a DataFrame.
    """
    
    def __init__(self, data_portal):
        """
        Initialize a new PipelineEngine.
        
        Parameters
        ----------
        data_portal : DataPortal
            The data portal used for accessing market data
        """
        self.data_portal = data_portal
        self.logger = logging.getLogger('vegas.pipeline.engine')
        
    def run_pipeline(self, pipeline: Pipeline, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pl.DataFrame:
        """
        Compute a pipeline for a date range.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to compute
        start_date : datetime or str
            Start date for the computation
        end_date : datetime or str
            End date for the computation
            
        Returns
        -------
        pl.DataFrame
            A DataFrame containing the computed pipeline values
        """
        start_date = self._to_datetime(start_date, tz=self.data_portal.timezone)
        end_date = self._to_datetime(end_date, tz=self.data_portal.timezone)
        
        max_window = self._get_max_window_length(pipeline)
        
        # Retrieve market data for the lookback window up to current_dt explicitly
        end_dt = getattr(self.data_portal, "_current_dt", None)
        market_data = self.data_portal.history(
            assets=None,  # All assets
            fields=None,  # All fields
            bar_count=max_window,
            frequency=pipeline.frequency,
            end_dt=end_dt,
        )

        # Sanitize symbols early to prevent None/invalid entries from propagating
        try:
            if not market_data.is_empty():
                if 'symbol' in market_data.columns:
                    if market_data.get_column('symbol').dtype != pl.Utf8:
                        market_data = market_data.with_columns(pl.col('symbol').cast(pl.Utf8))
                    market_data = market_data.filter(pl.col('symbol').is_not_null() & (pl.col('symbol') != ""))
        except Exception:
            pass
        
        if market_data.is_empty():
            self.logger.warning(f"No market data found between {start_date.date()} and {end_date.date()}")
            return pl.DataFrame()

        # Build the expressions for all terms
        expressions = [
            term.to_expression().alias(name)
            for name, term in pipeline.columns.items()
        ]

        # Add the screen expression if it exists
        if pipeline.screen:
            expressions.append(pipeline.screen.to_expression().alias('_screen_mask'))

        # Run the expressions over the market data
        result = market_data.lazy().with_columns(expressions).collect(engine='streaming')

        # Final sanitization of result symbols
        try:
            if not result.is_empty() and 'symbol' in result.columns:
                if result.get_column('symbol').dtype != pl.Utf8:
                    result = result.with_columns(pl.col('symbol').cast(pl.Utf8))
                result = result.filter(pl.col('symbol').is_not_null() & (pl.col('symbol') != ""))
        except Exception:
            pass

        # Apply the screen if it exists
        if pipeline.screen:
            result = result.filter(pl.col('_screen_mask')).drop('_screen_mask')

        # Keep explicit columns; put date and symbol first
        ordered_cols = ['timestamp', 'symbol'] + list(pipeline.columns.keys())
        result = result.select(ordered_cols)
        
        return result
    
    def _get_max_window_length(self, pipeline: Pipeline) -> int:
        """
        Get the maximum window length required by any term in the pipeline.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to analyze
            
        Returns
        -------
        int
            The maximum window length
        """
        max_window = 1  # Default minimum
        
        # Check each column
        for term in pipeline.columns.values():
            max_window = max(max_window, term.window_length)
        
        # Check screen
        if pipeline.screen is not None:
            max_window = max(max_window, pipeline.screen.window_length)
        
        return max_window
    
    def _to_datetime(self, value: Union[str, datetime, date], tz: str) -> datetime:
        """
        Convert supported inputs to a Python datetime object.
        Accepts datetime, date, or ISO-like strings.
        """
        if isinstance(value, datetime):
            return value.astimezone(pytz.timezone(tz))
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time()).astimezone(pytz.timezone(tz))
        if isinstance(value, str):
            # Try multiple common formats; fallback to fromisoformat
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    from datetime import datetime as _dt
                    return _dt.strptime(value, fmt)
                except Exception:
                    pass
            # Last resort
            try:
                from datetime import datetime as _dt
                return _dt.fromisoformat(value)
            except Exception:
                self.logger.error(f"Unable to parse date string: {value}")
                raise
        raise TypeError(f"Unsupported date type: {type(value)}")

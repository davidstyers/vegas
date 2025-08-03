"""Pipeline engine implementation for the Vegas backtesting system.

This module defines the PipelineEngine class that computes pipelines.
"""
from typing import Dict, List, Optional, Union, Any
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import logging

from vegas.pipeline.pipeline import Pipeline
from vegas.pipeline.terms import Term, Factor, Filter, Classifier


class PipelineEngine:
    """
    An engine for computing Pipelines.
    
    The PipelineEngine is responsible for executing pipeline computations
    and returning the results as a DataFrame.
    """
    
    def __init__(self, data_layer):
        """
        Initialize a new PipelineEngine.
        
        Parameters
        ----------
        data_layer : DataLayer
            The data layer used for accessing market data
        """
        self.data_layer = data_layer
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
        start_date = self._to_datetime(start_date)
        end_date = self._to_datetime(end_date)

        if start_date == end_date:
            trading_days = pl.Series([start_date])
        else:
            trading_days = self.data_layer.get_trading_days(start_date, end_date)
        
        if trading_days.is_empty():
            self.logger.warning(f"No trading days found between {start_date.date()} and {end_date.date()}")
            return pl.DataFrame()
        
        # Collect daily results as polars DataFrames
        all_results: List[pl.DataFrame] = []
        
        # Process each day
        for day in trading_days:
            day_results = self._compute_pipeline_for_day(pipeline, day.date())
            if day_results is not None and day_results.height > 0:
                # Add date column if not already present
                if 'date' not in day_results.columns:
                    day_results = day_results.with_columns(pl.lit(day).alias('date'))
                all_results.append(day_results)
        
        # Combine all daily results
        if not all_results:
            self.logger.warning("No results computed for any day in the date range")
            return pl.DataFrame()
        
        result = pl.concat(all_results, how="vertical")
        
        # Keep explicit columns; put date and symbol first
        ordered_cols = ['date', 'symbol'] + [c for c in result.columns if c not in ('date', 'symbol')]
        result = result.select(ordered_cols)
        
        return result
    
    def _compute_pipeline_for_day(self, pipeline: Pipeline, day: datetime.date) -> pl.DataFrame:
        """
        Compute a pipeline for a single day.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to compute
        day : datetime
            The day to compute the pipeline for
            
        Returns
        -------
        pl.DataFrame
            A DataFrame containing the computed pipeline values for the day
        """
        try:
            # Get data for the day and any lookback window
            max_window = self._get_max_window_length(pipeline)
            trading_days = self.data_layer.get_all_trading_days()
            lookback_start = (
                trading_days.filter(trading_days <= day)[-max_window:][0]
                if trading_days.filter(trading_days <= day).len() >= max_window
                else None
            )

            prev_days = trading_days.filter(trading_days < day)
            prev_day = prev_days[-1] if prev_days.len() > 0 else None
            
            # Get market data for the computation
            market_data = self.data_layer.get_data_for_backtest(lookback_start, day)
            
            if market_data.is_empty():
                self.logger.warning(f"No market data found for {day}")
                return pl.DataFrame()
            
            # Get the universe of assets for the previous day
            assets = self.data_layer.get_universe(prev_day)
            
            if not assets or len(assets) == 0:
                self.logger.warning(f"No assets found for {day}")
                # Try to get assets from the market data as a fallback
                if 'symbol' in market_data.columns:
                    assets = market_data.select(pl.col('symbol').unique().sort()).to_series().to_list()
                    if assets:
                        self.logger.info(f"Using {len(assets)} assets from market data")
                    else:
                        return pl.DataFrame()
                else:
                    return pl.DataFrame()
            
            # Convert assets to strings to ensure consistency
            assets = [str(asset) for asset in assets]
            
            # Compute each column
            col_names = list(pipeline.columns.keys())
            results_matrix: Dict[str, np.ndarray] = {}
            for name, term in pipeline.columns.items():
                try:
                    term_result = self._compute_term(term, day, assets, market_data)
                    results_matrix[name] = term_result
                except Exception as e:
                    self.logger.error(f"Error computing term {name}: {e}")
                    results_matrix[name] = np.full(len(assets), np.nan)
            
            # Build polars DataFrame
            data_dict: Dict[str, Any] = {name: results_matrix[name] for name in col_names}
            data_dict['symbol'] = assets
            result_df = pl.DataFrame(data_dict)
            
            # Apply screen if present
            if pipeline.screen is not None:
                try:
                    screen_result = self._compute_term(pipeline.screen, day, assets, market_data)
                    mask = np.asarray(screen_result, dtype=bool)
                    if mask.shape[0] != result_df.height:
                        self.logger.error("Screen result length does not match result rows; skipping screen")
                    else:
                        result_df = result_df.with_columns(pl.Series('_screen_mask', mask))
                        result_df = result_df.filter(pl.col('_screen_mask')).drop('_screen_mask')
                except Exception as e:
                    self.logger.error(f"Error applying screen: {e}")
            
            return result_df
        
        except Exception as e:
            self.logger.error(f"Error computing pipeline for {day}: {e}")
            return pl.DataFrame()
    
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
    
    def _compute_term(self, term: Term, day: datetime, assets: List[str], market_data: pl.DataFrame) -> np.ndarray:
        """
        Compute a single term for the given day and assets.
        
        Parameters
        ----------
        term : Term
            The term to compute
        day : datetime
            The day to compute the term for
        assets : list of str
            The assets to compute the term for
        market_data : pl.DataFrame
            The market data to use for computation
            
        Returns
        -------
        np.ndarray
            An array of computed values
        """
        # Create output array
        out = np.full(len(assets), term.missing_value, dtype=term.dtype)
        
        # If term has inputs, compute those first
        input_arrays: List[np.ndarray] = []
        if hasattr(term, 'inputs') and term.inputs:
            for input_term in term.inputs:
                if isinstance(input_term, Term):
                    input_result = self._compute_term(input_term, day, assets, market_data)
                    input_arrays.append(input_result)
                elif isinstance(input_term, str):
                    # String inputs are column names in market_data
                    window_data = self._get_window_data(market_data, input_term, day, term.window_length)
                    if window_data is not None:
                        input_arrays.append(window_data)
                    else:
                        return np.full(len(assets), term.missing_value, dtype=term.dtype)
        
        # Call the term's compute method
        try:
            term.compute(day, assets, out, *input_arrays)
            return out
        except Exception as e:
            self.logger.error(f"Error computing term {term}: {e}")
            return np.full(len(assets), term.missing_value, dtype=term.dtype)
    
    def _get_window_data(self, market_data: pl.DataFrame, column: str, day: datetime, window_length: int) -> Optional[np.ndarray]:
        """
        Get window_length days of data for a column up to and including day.
        
        Parameters
        ----------
        market_data : pl.DataFrame
            The market data
        column : str
            The column to get data for
        day : datetime
            The last day to include
        window_length : int
            The number of days to include
            
        Returns
        -------
        np.ndarray or None
            Array of shape (window_length, n_assets) or None if data not available
        """
        if column not in market_data.columns:
            self.logger.warning(f"Column '{column}' not found in market data")
            return None
        
        if 'timestamp' not in market_data.columns:
            self.logger.warning("Column 'timestamp' not found in market data")
            return None
        if 'symbol' not in market_data.columns:
            self.logger.warning("Column 'symbol' not found in market data")
            return None
        
        md = market_data
        # Ensure timestamp is datetime for comparison
        ts_dtype = md.schema.get('timestamp')
        if ts_dtype == pl.Utf8:
            md = md.with_columns(pl.col('timestamp').str.strptime(pl.Datetime, strict=False))
        elif ts_dtype == pl.Date:
            md = md.with_columns(pl.col('timestamp').cast(pl.Datetime))
        
        filtered = md.filter(pl.col('timestamp') <= pl.lit(day))
        
        if filtered.is_empty():
            return None
        
        # Determine consistent symbol ordering
        symbols = (
            filtered.select(pl.col('symbol').unique())
            .to_series()
            .to_list()
        )
        symbols = sorted([str(s) for s in symbols if s is not None])
        
        if not symbols:
            return None
        
        per_symbol_arrays: Dict[str, np.ndarray] = {}
        
        for sym in symbols:
            sdf = filtered.filter(pl.col('symbol') == sym).sort('timestamp')
            if sdf.is_empty():
                per_symbol_arrays[sym] = np.full(window_length, np.nan, dtype=float)
                continue
            vals = sdf.select(pl.col(column)).to_series().to_numpy()
            if vals.shape[0] >= window_length:
                arr = vals[-window_length:]
            else:
                pad = np.full(window_length - vals.shape[0], np.nan, dtype=float)
                arr = np.concatenate([pad, vals.astype(float, copy=False)])
            per_symbol_arrays[sym] = arr
        
        if not per_symbol_arrays:
            return None
        
        mat = np.column_stack([per_symbol_arrays[s] for s in symbols]).astype(float, copy=False)
        return mat

    def _to_polars_df(self, df_any: Any) -> pl.DataFrame:
        """
        Normalize inbound data to polars DataFrame.
        """
        if isinstance(df_any, pl.DataFrame):
            return df_any
        if df_any is None:
            return pl.DataFrame()
        # Try constructing from dict/records or sequence
        try:
            return pl.DataFrame(df_any)
        except Exception:
            self.logger.error("Unable to convert market_data to polars DataFrame")
            return pl.DataFrame()

    def _to_datetime(self, value: Union[str, datetime]) -> datetime:
        """
        Convert supported inputs to a Python datetime without pandas.
        Accepts datetime or ISO-like strings.
        """
        if isinstance(value, datetime):
            return value
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

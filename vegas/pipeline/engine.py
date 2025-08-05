"""Pipeline engine implementation for the Vegas backtesting system.

This module defines the PipelineEngine class that computes pipelines.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
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
        Compute a pipeline for a single day using vectorized Polars windows.
        Assumes inbound market data is already a Polars DataFrame.
        """
        try:
            # Determine lookback bounds from maximum window across the pipeline
            max_window = self._get_max_window_length(pipeline)
            trading_days = self.data_layer.get_all_trading_days()
            eligible = trading_days.filter(trading_days <= day)
            lookback_start = eligible[-max_window:][0] if eligible.len() >= max_window else None

            prev_days = trading_days.filter(trading_days < day)
            prev_day = prev_days[-1] if prev_days.len() > 0 else None

            # Retrieve market data for the lookback window
            market_data = self.data_layer.get_data_for_backtest(lookback_start, day)
            if market_data.is_empty():
                self.logger.warning(f"No market data found for {day}")
                return pl.DataFrame()

            # Resolve universe: prefer engine's universe from prev_day, fallback to market_data
            uni = self.data_layer.get_universe(prev_day)
            if not uni or len(uni) == 0:
                if 'symbol' in market_data.columns:
                    assets_order = (
                        market_data.filter(pl.col('timestamp') <= pl.lit(day))
                        .select(pl.col('symbol').cast(pl.Utf8).unique())
                        .to_series()
                        .drop_nulls()
                        .to_list()
                    )
                    assets_order = sorted([str(a) for a in assets_order])
                    if not assets_order:
                        return pl.DataFrame()
                    self.logger.info(f"Using {len(assets_order)} assets from market data")
                else:
                    return pl.DataFrame()
            else:
                assets_order = sorted([str(a) for a in uni])

            # Compute each column
            col_names = list(pipeline.columns.keys())
            results_matrix: Dict[str, np.ndarray] = {}
            for name, term in pipeline.columns.items():
                try:
                    self.logger.info(f"Computing term {name} for {day}")
                    term_result = self._compute_term(term, day, assets_order, market_data)
                    results_matrix[name] = term_result
                except Exception as e:
                    self.logger.error(f"Error computing term {name}: {e}")
                    results_matrix[name] = np.full(len(assets_order), np.nan)

            # Assemble result DataFrame
            data_dict: Dict[str, Any] = {name: results_matrix[name] for name in col_names}
            data_dict['symbol'] = assets_order
            result_df = pl.DataFrame(data_dict)

            # Apply screen if present
            if pipeline.screen is not None:
                try:
                    # Compute screen and coerce to strict boolean mask with correct length.
                    raw_mask = self._compute_term(pipeline.screen, day, assets_order, market_data)
                    mask = np.asarray(raw_mask)
                    # If object dtype leaked through Filters, convert elementwise to bool
                    if mask.dtype == object:
                        mask = np.array([bool(x) for x in mask.ravel()], dtype=bool)
                    else:
                        mask = mask.astype(bool, copy=False).ravel()
                    # Validate length and broadcast scalar if needed
                    if mask.shape[0] != len(assets_order):
                        if mask.shape[0] == 1:
                            mask = np.full(len(assets_order), bool(mask[0]), dtype=bool)
                        else:
                            self.logger.error("Screen result length does not match result rows; skipping screen")
                            mask = None
                    if mask is not None:
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
        Compute a single term for the given day and assets using vectorized windows for raw inputs.
        """
        # Ensure dtype is respected; Filters may request object dtype internally but engine expects arrays.
        # Build output with the exact dtype requested by the term.
        out = np.full(len(assets), term.missing_value, dtype=term.dtype)

        # Resolve inputs
        input_arrays: List[np.ndarray] = []
        if hasattr(term, 'inputs') and term.inputs:
            for input_term in term.inputs:
                if isinstance(input_term, Term):
                    input_arrays.append(self._compute_term(input_term, day, assets, market_data))
                elif isinstance(input_term, str):
                    wlen = getattr(term, 'window_length', 1) or 1
                    arr = self._get_window_data(market_data, input_term, day, wlen, assets)
                    if arr is None:
                        return np.full(len(assets), term.missing_value, dtype=term.dtype)
                    if arr.dtype != np.float64:
                        arr = arr.astype(np.float64, copy=False)
                    input_arrays.append(arr)

        try:
            term.compute(day, assets, out, *input_arrays)
            return out
        except Exception as e:
            self.logger.error(f"Error computing term {term}: {e}")
            return np.full(len(assets), term.missing_value, dtype=term.dtype)
    
    def _get_window_data(self, market_data: pl.DataFrame, column: str, day: datetime, window_length: int, assets_order: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """
        Vectorized window retrieval for a raw input column using a single Polars pivot.

        Returns float64 ndarray of shape (window_length, n_assets) aligned to assets_order.
        """
        if column not in market_data.columns:
            self.logger.warning(f"Column '{column}' not found in market data")
            return None
        if 'timestamp' not in market_data.columns or 'symbol' not in market_data.columns:
            self.logger.warning("Required columns 'timestamp' or 'symbol' not found in market data")
            return None

        # Derive deterministic assets order if not provided
        if not assets_order:
            symbols = (
                market_data.filter(pl.col('timestamp') <= pl.lit(day))
                .select(pl.col('symbol').cast(pl.Utf8).unique())
                .to_series()
                .drop_nulls()
                .to_list()
            )
            assets_order = sorted([str(s) for s in symbols])

        # Filter and pivot
        fdf = market_data.filter(
            (pl.col('timestamp') <= pl.lit(day)) & (pl.col('symbol').is_in(assets_order))
        )
        if fdf.is_empty():
            return np.full((window_length, len(assets_order)), np.nan, dtype=np.float64)

        fdf = fdf.select(
            pl.col('timestamp').cast(pl.Datetime),
            pl.col('symbol').cast(pl.Utf8),
            pl.col(column).cast(pl.Float64),
        ).sort(['timestamp', 'symbol'])

        pivot = fdf.pivot(index='timestamp', columns='symbol', values=column, aggregate_function='first')

        # Align pivot columns to assets_order
        symbol_cols = [c for c in pivot.columns if c != 'timestamp']
        missing = [s for s in assets_order if s not in symbol_cols]
        if missing:
            pivot = pivot.with_columns([pl.lit(None, dtype=pl.Float64).alias(s) for s in missing])
        pivot = pivot.select(['timestamp'] + assets_order)

        # Tail and convert
        tail = pivot.tail(window_length)
        mat = tail.select(assets_order).to_numpy()
        if mat.dtype != np.float64:
            mat = mat.astype(np.float64, copy=False)

        # Top-pad if needed
        if mat.shape[0] < window_length:
            pad_rows = window_length - mat.shape[0]
            pad = np.full((pad_rows, mat.shape[1]), np.nan, dtype=np.float64)
            mat = np.vstack([pad, mat])

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

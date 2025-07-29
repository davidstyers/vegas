"""Pipeline engine implementation for the Vegas backtesting system.

This module defines the PipelineEngine class that computes pipelines.
"""
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
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
        
    def run_pipeline(self, pipeline: Pipeline, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
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
        pd.DataFrame
            A DataFrame containing the computed pipeline values
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        self.logger.info(f"Computing pipeline from {start_date.date()} to {end_date.date()}")
        
        # Get the trading days in the date range
        trading_days = self.data_layer.get_trading_days(start_date, end_date)
        
        if len(trading_days) == 0:
            self.logger.warning(f"No trading days found between {start_date.date()} and {end_date.date()}")
            return pd.DataFrame()
        
        # Initialize the result DataFrame
        all_results = []
        
        # Process each day
        for day in trading_days:
            day_results = self._compute_pipeline_for_day(pipeline, day)
            if day_results is not None and not day_results.empty:
                # Add date column if not already present
                if 'date' not in day_results.columns:
                    day_results['date'] = day
                all_results.append(day_results)
        
        # Combine all daily results
        if not all_results:
            self.logger.warning("No results computed for any day in the date range")
            return pd.DataFrame()
        
        result = pd.concat(all_results, ignore_index=True)
        
        # Set the index to be a MultiIndex of (date, asset)
        if 'symbol' in result.columns and 'date' in result.columns:
            result.set_index(['date', 'symbol'], inplace=True)
        
        return result
    
    def _compute_pipeline_for_day(self, pipeline: Pipeline, day: datetime) -> pd.DataFrame:
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
        pd.DataFrame
            A DataFrame containing the computed pipeline values for the day
        """
        try:
            # Get data for the day and any lookback window
            max_window = self._get_max_window_length(pipeline)
            lookback_start = day - timedelta(days=max_window*2)  # Allow extra days for weekends/holidays
            
            # Get market data for the computation
            market_data = self.data_layer.get_data_for_backtest(lookback_start, day)
            
            if market_data.empty:
                self.logger.warning(f"No market data found for {day.date()}")
                return pd.DataFrame()
            
            # Get the universe of assets for the day
            assets = self.data_layer.get_universe(day)
            
            if not assets or len(assets) == 0:
                self.logger.warning(f"No assets found for {day.date()}")
                # Try to get assets from the market data as a fallback
                if 'symbol' in market_data.columns:
                    assets = market_data['symbol'].unique().tolist()
                    if assets:
                        self.logger.info(f"Using {len(assets)} assets from market data")
                    else:
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()
            
            # Convert assets to strings to ensure consistency
            assets = [str(asset) for asset in assets]
            
            # Initialize results dictionary
            results = {}
            
            # Compute each column
            for name, term in pipeline.columns.items():
                try:
                    # Compute the term
                    term_result = self._compute_term(term, day, assets, market_data)
                    results[name] = term_result
                except Exception as e:
                    self.logger.error(f"Error computing term {name}: {e}")
                    # Use NaN for this term
                    results[name] = np.full(len(assets), np.nan)
            
            # Create DataFrame from results
            result_df = pd.DataFrame(results)
            result_df['symbol'] = assets
            
            # Apply screen if present
            if pipeline.screen is not None:
                try:
                    screen_result = self._compute_term(pipeline.screen, day, assets, market_data)
                    # Only keep rows where the screen is True
                    result_df = result_df[screen_result.astype(bool)]
                except Exception as e:
                    self.logger.error(f"Error applying screen: {e}")
                    # If screen fails, return all results
            
            return result_df
        
        except Exception as e:
            self.logger.error(f"Error computing pipeline for {day.date()}: {e}")
            return pd.DataFrame()
    
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
    
    def _compute_term(self, term: Term, day: datetime, assets: List[str], market_data: pd.DataFrame) -> np.ndarray:
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
        market_data : pd.DataFrame
            The market data to use for computation
            
        Returns
        -------
        np.ndarray
            An array of computed values
        """
        # Create output array
        out = np.full(len(assets), term.missing_value, dtype=term.dtype)
        
        # For now, this is a simplified implementation that doesn't handle complex terms with inputs
        # We'll need to expand this to handle the full dependency graph of terms
        
        # If term has inputs, compute those first
        input_arrays = []
        if hasattr(term, 'inputs') and term.inputs:
            for input_term in term.inputs:
                if isinstance(input_term, Term):
                    input_result = self._compute_term(input_term, day, assets, market_data)
                    input_arrays.append(input_result)
                elif isinstance(input_term, str):
                    # String inputs are column names in market_data
                    # Get the data for the window length
                    window_data = self._get_window_data(market_data, input_term, day, term.window_length)
                    if window_data is not None:
                        input_arrays.append(window_data)
                    else:
                        # If we couldn't get data, return array of missing values
                        return np.full(len(assets), term.missing_value, dtype=term.dtype)
        
        # Call the term's compute method
        try:
            term.compute(day, assets, out, *input_arrays)
            return out
        except Exception as e:
            self.logger.error(f"Error computing term {term}: {e}")
            return np.full(len(assets), term.missing_value, dtype=term.dtype)
    
    def _get_window_data(self, market_data: pd.DataFrame, column: str, day: datetime, window_length: int) -> Optional[np.ndarray]:
        """
        Get window_length days of data for a column up to and including day.
        
        Parameters
        ----------
        market_data : pd.DataFrame
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
        
        # Filter data to be on or before the given day
        filtered_data = market_data[market_data['timestamp'] <= day]
        
        if filtered_data.empty:
            return None
        
        # Get the unique symbols in the data
        symbols = sorted(filtered_data['symbol'].unique())
        
        # Create a dictionary to hold data for each symbol
        symbol_data = {}
        
        # Group by symbol and get the data for each
        for symbol in symbols:
            symbol_data[symbol] = []
            
            # Get data for this symbol
            symbol_df = filtered_data[filtered_data['symbol'] == symbol]
            
            if not symbol_df.empty:
                # Sort by timestamp
                symbol_df = symbol_df.sort_values('timestamp')
                
                # Get the last window_length rows or pad with NaNs
                values = symbol_df[column].values
                if len(values) >= window_length:
                    symbol_data[symbol] = values[-window_length:]
                else:
                    # Pad with NaNs at the beginning
                    padding = [np.nan] * (window_length - len(values))
                    symbol_data[symbol] = np.array(padding + list(values))
            else:
                # No data for this symbol, use NaNs
                symbol_data[symbol] = np.full(window_length, np.nan)
        
        # Convert to a 2D array (window_length x num_symbols)
        result = np.array([symbol_data[symbol] for symbol in symbols]).T
        
        return result 
"""Statistical factors for the Vegas pipeline system.

This module defines factors for statistical transformations like Z-Score and ranking.
"""
import numpy as np
from vegas.pipeline.factors.custom import CustomFactor
from vegas.pipeline.terms import Term


class ZScore(CustomFactor):
    """
    Factor producing Z-scores for each day's cross-section.
    
    Z-scores are computed using:
    
    z = (x - mean(x)) / std(x)
    
    where x is the input data on a given day.
    
    Parameters
    ----------
    inputs : list, optional
        A list of data inputs to use in compute.
    window_length : int, optional
        The number of rows of data to pass to compute.
    mask : Filter, optional
        A Filter defining values to compute.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, mask=None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def compute(self, today, assets, out, data):
        """
        Compute Z-scores for the input data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Input data to compute Z-scores from (typically from self.term)
        """
        # Get the last row of data (most recent values)
        if data.ndim > 1:
            row = data[-1]
        else:
            row = data
            
        # Calculate mean and standard deviation
        mean = np.nanmean(row)
        std = np.nanstd(row)
        
        # Handle edge cases - if std is 0 or all data is NaN
        if std == 0 or np.isnan(std):
            # If we can't compute meaningful Z-scores, output NaNs
            out[:] = np.nan
        else:
            # Calculate Z-scores
            out[:] = (row - mean) / std


class Rank(CustomFactor):
    """
    Factor representing the sorted rank of each column within each row.
    
    Parameters
    ----------
    term : Term
        The term to rank
    method : {'ordinal', 'min', 'max', 'dense', 'average'}, optional
        The method used to assign ranks to tied elements.
    ascending : bool, optional
        Whether to rank in ascending or descending order.
    mask : Filter, optional
        A Filter representing assets to consider when computing ranks.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, method='ordinal', ascending=True, mask=None):
        self.term = term
        self.method = method
        self.ascending = ascending
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def compute(self, today, assets, out, data):
        """
        Compute ranks for the input data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Input data to compute ranks from (typically from self.term)
        """
        # Get the last row of data (most recent values)
        if data.ndim > 1:
            row = data[-1]
        else:
            row = data
        
        # Handle NaN values - they should be ranked last
        mask = np.isfinite(row)
        
        # Create array for ranks (initialize with NaN for assets that should be excluded)
        ranks = np.full_like(row, np.nan)
        
        # Only rank finite values
        values = row[mask]
        if len(values) > 0:
            # Calculate ranks for finite values
            method_map = {
                'ordinal': 'ordinal',
                'min': 'min',
                'max': 'max',
                'dense': 'dense',
                'average': 'average'
            }
            
            # Use numpy's rankdata function with the specified method
            from scipy.stats import rankdata
            
            # Adjust the order based on ascending parameter
            if self.ascending:
                ranked = rankdata(values, method=method_map[self.method])
            else:
                # For descending, reverse the ranks
                ranked = len(values) + 1 - rankdata(values, method=method_map[self.method])
                
            # Put the ranks back in the original positions
            ranks[mask] = ranked
            
        # Set output
        out[:] = ranks


class Percentile(CustomFactor):
    """
    Factor representing percentiles of data.
    
    Parameters
    ----------
    term : Term
        The term to compute percentiles for
    mask : Filter, optional
        A Filter representing assets to consider when computing percentiles.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, mask=None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def compute(self, today, assets, out, data):
        """
        Compute percentiles for the input data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Input data to compute percentiles from (typically from self.term)
        """
        # Get the last row of data (most recent values)
        if data.ndim > 1:
            row = data[-1]
        else:
            row = data
        
        # Handle NaN values
        mask = np.isfinite(row)
        
        # Create array for percentiles (initialize with NaN for assets that should be excluded)
        percentiles = np.full_like(row, np.nan)
        
        # Only compute percentiles for finite values
        values = row[mask]
        if len(values) > 0:
            from scipy.stats import percentileofscore
            
            # Calculate percentile for each value
            for i in np.where(mask)[0]:
                percentiles[i] = percentileofscore(values, row[i]) / 100.0
            
        # Set output
        out[:] = percentiles 
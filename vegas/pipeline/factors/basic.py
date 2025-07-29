"""Basic built-in factors for the Vegas pipeline system.

This module defines common basic factors like Returns and moving averages.
"""
import numpy as np
import pandas as pd
from vegas.pipeline.factors.custom import CustomFactor


class Returns(CustomFactor):
    """
    Factor computing the percentage change in price over a given window.
    """
    inputs = ['close']
    window_length = 2  # Default to daily returns
    
    def compute(self, today, assets, out, closes):
        """
        Calculate returns from price data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        closes : np.array
            Price data arrays to compute returns from
        """
        try:
            # Ensure data is the right shape
            if closes.ndim == 1:
                closes = closes.reshape(-1, 1)
            
            # Calculate returns
            out[:] = (closes[-1] - closes[0]) / closes[0]
        except Exception as e:
            # If calculation fails, fill with NaN
            out[:] = np.nan


class SimpleMovingAverage(CustomFactor):
    """
    Factor computing a simple moving average of a data input.
    
    Examples
    --------
    Calculate a 10-day SMA of closing prices:
    
    >>> sma = SimpleMovingAverage(inputs=['close'], window_length=10)
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day moving average
    
    def compute(self, today, assets, out, data):
        """
        Calculate simple moving average.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Data to compute moving average from
        """
        try:
            # Ensure data is the right shape
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Calculate mean along axis 0 (time)
            out[:] = np.nanmean(data, axis=0)
        except Exception as e:
            # If calculation fails, fill with NaN
            out[:] = np.nan


class ExponentialWeightedMovingAverage(CustomFactor):
    """
    Factor computing an exponentially-weighted moving average of a data input.
    
    Parameters
    ----------
    inputs : list, optional
        A list of data inputs to use in compute.
    window_length : int, optional
        The number of rows of data to pass to compute.
    decay_rate : float, optional
        The rate at which weights decrease. Higher values means more weight
        for more recent observations.
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day moving average
    
    def __init__(self, inputs=None, window_length=None, decay_rate=0.5, mask=None):
        self.decay_rate = decay_rate
        super().__init__(inputs=inputs, window_length=window_length, mask=mask)
    
    def compute(self, today, assets, out, data):
        """
        Calculate exponentially-weighted moving average.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Data to compute moving average from
        """
        try:
            # Ensure data is the right shape
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                
            # Create weights that exponentially decay based on the provided rate
            # The weights array will have shape (window_length,)
            weights = np.power(self.decay_rate, np.arange(self.window_length - 1, -1, -1))
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Weighted average (handle NaNs by setting them to 0 and renormalizing weights)
            weighted_data = data * weights.reshape(-1, 1)  # Broadcasting weights across assets
            
            # Handle NaNs by using nansum and rescaling weights
            mask = ~np.isnan(data)
            valid_weights = np.sum(weights.reshape(-1, 1) * mask, axis=0)  # Sum of weights for non-NaN values
            out[:] = np.nansum(weighted_data, axis=0) / valid_weights
        except Exception as e:
            # If calculation fails, fill with NaN
            out[:] = np.nan


class VWAP(CustomFactor):
    """
    Factor computing Volume Weighted Average Price.
    
    Volume Weighted Average Price (VWAP) is the ratio of the value traded to total volume
    traded over a particular time horizon (usually one day).
    """
    inputs = ['close', 'volume']
    window_length = 1  # Default to single day VWAP
    
    def compute(self, today, assets, out, closes, volumes):
        """
        Calculate VWAP from price and volume data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        closes : np.array
            Price data
        volumes : np.array
            Volume data
        """
        try:
            # Ensure data is the right shape
            if closes.ndim == 1:
                closes = closes.reshape(-1, 1)
            if volumes.ndim == 1:
                volumes = volumes.reshape(-1, 1)
                
            # Calculate the product of price and volume
            value_traded = closes * volumes
            
            # Calculate VWAP
            with np.errstate(divide='ignore', invalid='ignore'):  # Ignore divide by zero warnings
                out[:] = np.nansum(value_traded, axis=0) / np.nansum(volumes, axis=0)
            
            # Replace any NaNs or infinities with the simple average price
            invalid_mask = ~np.isfinite(out)
            if np.any(invalid_mask):
                # For assets with zero volume, use the simple average price
                out[invalid_mask] = np.nanmean(closes[:, invalid_mask], axis=0)
        except Exception as e:
            # If calculation fails, fill with NaN
            out[:] = np.nan


class StandardDeviation(CustomFactor):
    """
    Factor computing the standard deviation of a data input over a window.
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day window
    
    def compute(self, today, assets, out, data):
        """
        Calculate standard deviation.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        data : np.array
            Data to compute standard deviation from
        """
        try:
            # Ensure data is the right shape
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Calculate standard deviation along axis 0 (time)
            out[:] = np.nanstd(data, axis=0)
        except Exception as e:
            # If calculation fails, fill with NaN
            out[:] = np.nan 
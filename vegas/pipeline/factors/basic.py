"""Basic built-in factors for the Vegas pipeline system.

This module defines common basic factors like Returns and moving averages.
"""
import polars as pl
from vegas.pipeline.factors.custom import CustomFactor


class Returns(CustomFactor):
    """
    Factor computing the percentage change in price over a given window.
    """
    inputs = ['close']
    window_length = 2  # Default to daily returns
    
    def to_expression(self) -> pl.Expr:
        """
        Calculate returns from price data.
        """
        return pl.col(self.inputs[0]).pct_change(n=self.window_length - 1).over('symbol')


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
    
    def to_expression(self) -> pl.Expr:
        """
        Calculate simple moving average.
        """
        return pl.col(self.inputs[0]).rolling_mean(self.window_length).over('symbol')


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
    
    def to_expression(self) -> pl.Expr:
        """
        Calculate exponentially-weighted moving average.
        """
        return pl.col(self.inputs[0]).ewm_mean(alpha=self.decay_rate, adjust=False).over('symbol')


class VWAP(CustomFactor):
    """
    Factor computing Volume Weighted Average Price.
    
    Volume Weighted Average Price (VWAP) is the ratio of the value traded to total volume
    traded over a particular time horizon (usually one day).
    """
    inputs = ['close', 'volume']
    window_length = 1  # Default to single day VWAP
    
    def to_expression(self) -> pl.Expr:
        """
        Calculate VWAP from price and volume data.
        """
        return (pl.col('close') * pl.col('volume')).sum().over('symbol') / pl.col('volume').sum().over('symbol')


class StandardDeviation(CustomFactor):
    """
    Factor computing the standard deviation of a data input over a window.
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day window
    
    def to_expression(self) -> pl.Expr:
        """
        Calculate standard deviation.
        """
        return pl.col(self.inputs[0]).rolling_std(self.window_length).over('symbol')
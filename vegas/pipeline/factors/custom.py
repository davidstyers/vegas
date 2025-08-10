"""Custom factor implementation for the Vegas pipeline system.

This module defines the CustomFactor class that allows users to create their own factors.
"""
from vegas.pipeline.terms import Factor
import polars as pl


class CustomFactor(Factor):
    """
    Base class for user-defined Factors.
    
    CustomFactors allow users to easily define their own Factors
    by implementing a to_expression method that returns a Polars expression.
    
    Examples
    --------
    Define a simple moving average factor:
    
    >>> class SimpleMovingAverage(CustomFactor):
    ...     inputs = ['close']
    ...     window_length = 10
    ...     def to_expression(self):
    ...         return pl.col(self.inputs[0]).rolling_mean(self.window_length).over('symbol')
    ...
    >>> sma_10 = SimpleMovingAverage()
    """
    
    inputs = None
    window_length = None
    
    def __init__(self, inputs=None, window_length=None, mask=None, **kwargs):
        """
        Initialize a CustomFactor.
        
        Parameters
        ----------
        inputs : list, optional
            A list of data inputs to use in compute.
        window_length : int, optional
            The number of rows of data to pass to compute.
        mask : Filter, optional
            A Filter defining values to compute.
        **kwargs
            Additional keyword arguments.
        """
        if inputs is None:
            if self.inputs is None:
                raise ValueError(
                    f"{type(self).__name__} requires 'inputs' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            inputs = self.inputs
            
        if window_length is None:
            if self.window_length is None:
                raise ValueError(
                    f"{type(self).__name__} requires 'window_length' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            window_length = self.window_length
        
        super().__init__(inputs=inputs, window_length=window_length, mask=mask)
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_expression(self) -> pl.Expr:
        raise NotImplementedError("CustomFactor subclasses must implement to_expression")
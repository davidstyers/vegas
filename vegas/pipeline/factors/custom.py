"""Custom factor implementation for the Vegas pipeline system.

This module defines the CustomFactor class that allows users to create their own factors.
"""
from typing import List, Optional, Union, Any, Type, Dict
import numpy as np
import pandas as pd
from vegas.pipeline.terms import Factor


class CustomFactor(Factor):
    """
    Base class for user-defined Factors.
    
    CustomFactors allow users to easily define their own Factors
    by implementing a compute method that operates on input data.
    
    Examples
    --------
    Define a simple moving average factor:
    
    >>> class SimpleMovingAverage(CustomFactor):
    ...     def compute(self, today, assets, out, closes):
    ...         out[:] = np.nanmean(closes, axis=0)
    ...
    >>> sma_10 = SimpleMovingAverage(inputs=['close'], window_length=10)
    """
    
    inputs = None
    window_length = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create and return a new factor instance.
        
        This method is used to handle class-specified default inputs and window_length.
        """
        inputs = kwargs.pop('inputs', None)
        if inputs is None:
            if cls.inputs is None:
                raise ValueError(
                    f"{cls.__name__} requires 'inputs' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            inputs = cls.inputs
            
        window_length = kwargs.pop('window_length', None)
        if window_length is None:
            if cls.window_length is None:
                raise ValueError(
                    f"{cls.__name__} requires 'window_length' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            window_length = cls.window_length
            
        return super().__new__(cls)
    
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
        # These have already been handled in __new__, so just pass to parent constructor
        inputs = inputs if inputs is not None else self.inputs
        window_length = window_length if window_length is not None else self.window_length
        
        super().__init__(inputs=inputs, window_length=window_length, mask=mask)
        
        for k, v in kwargs.items():
            setattr(self, k, v) 
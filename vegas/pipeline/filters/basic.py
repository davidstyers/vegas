"""Basic filters for the Vegas pipeline system.

This module defines common basic filters like StaticAssets and comparison operations.
"""
from typing import List, Union, Any
import numpy as np
from vegas.pipeline.terms import Filter, Term


class StaticAssets(Filter):
    """
    Filter for a static set of assets.
    
    Parameters
    ----------
    assets : list
        List of asset symbols to include in the filter.
    """
    def __init__(self, assets):
        self.assets = set(assets)
        super().__init__()
        
    def compute(self, today, assets, out, *inputs):
        """
        Determine which assets match the static list.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Not used in this filter
        """
        # Use np.isin instead of np.in1d (deprecated)
        # Convert assets to strings if they're not already
        assets_list = [str(a) for a in assets]
        assets_set = set(self.assets)
        
        # Compare each asset to the set of allowed assets
        for i, asset in enumerate(assets_list):
            out[i] = asset in assets_set


class BinaryCompare(Filter):
    """
    A Filter representing a binary comparison.
    
    Parameters
    ----------
    left : Term or scalar
        Left side of the comparison.
    right : Term or scalar
        Right side of the comparison.
    op : {'<', '<=', '==', '!=', '>=', '>'}
        The comparison operator.
    """
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        
        # Gather inputs from any Terms
        inputs = []
        if isinstance(left, Term):
            inputs.append(left)
        if isinstance(right, Term):
            inputs.append(right)
        
        # Use the maximum window_length of any input
        window_length = max([t.window_length for t in inputs]) if inputs else 1
        
        # Initialize with the gathered inputs and window_length
        super().__init__(inputs=inputs, window_length=window_length)
    
    def compute(self, today, assets, out, *inputs):
        """
        Apply the comparison operation.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Input arrays from left and/or right Term
        """
        # Get the values for left and right
        if len(inputs) == 0:
            # Both are scalars
            left_value = self.left
            right_value = self.right
        elif len(inputs) == 1:
            # One Term, one scalar
            if isinstance(self.left, Term):
                left_value = inputs[0]
                right_value = self.right
            else:
                left_value = self.left
                right_value = inputs[0]
        else:
            # Both are Terms
            left_value = inputs[0]
            right_value = inputs[1]
        
        # Apply the comparison operator
        if self.op == '<':
            out[:] = left_value < right_value
        elif self.op == '<=':
            out[:] = left_value <= right_value
        elif self.op == '==':
            out[:] = left_value == right_value
        elif self.op == '!=':
            out[:] = left_value != right_value
        elif self.op == '>=':
            out[:] = left_value >= right_value
        elif self.op == '>':
            out[:] = left_value > right_value
        else:
            raise ValueError(f"Unknown comparison operator: {self.op}")


class NotNaN(Filter):
    """
    A filter that returns True for values that are not NaN.
    
    Parameters
    ----------
    term : Term
        The term to check for NaN values.
    """
    def __init__(self, term):
        self.term = term
        super().__init__(inputs=[term], window_length=term.window_length)
    
    def compute(self, today, assets, out, data):
        """
        Identify values that are not NaN.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        data : np.array
            Input data from the term
        """
        out[:] = ~np.isnan(data) 
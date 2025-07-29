"""Base classes for Pipeline Terms.

This module defines the core building blocks for computations in the pipeline system.
"""
from typing import List, Optional, Union, Any
import numpy as np
import pandas as pd


class Term:
    """
    Base class for all Pipeline computations.
    
    Terms are the core building blocks for pipeline expressions.
    """
    def __init__(self, 
                 inputs: Optional[List['Term']] = None, 
                 window_length: int = 1, 
                 mask: Optional['Filter'] = None):
        """
        Initialize a Term with inputs and parameters.
        
        Parameters
        ----------
        inputs : list of Terms, optional
            Terms whose outputs should be provided to this Term's compute function
        window_length : int, optional
            Number of rows of historical data to load for each compute
        mask : Filter, optional
            A Filter defining values to compute
        """
        self.inputs = inputs or []
        self.window_length = window_length
        self.mask = mask
        self.name = None
        self.dtype = np.float64
        self.missing_value = np.nan
        
    def compute(self, today: pd.Timestamp, assets: np.ndarray, out: np.ndarray, *inputs) -> None:
        """
        Calculate values for this Term.
        
        This method is called with raw data for all inputs and must write
        output values into the `out` array.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array[int64]
            The assets for which values are requested
        out : np.array
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Raw data arrays for any inputs to this Term
        """
        raise NotImplementedError("Term subclasses must implement compute")
    
    def __lt__(self, other) -> 'Filter':
        """Binary operator <"""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '<')

    def __le__(self, other) -> 'Filter':
        """Binary operator <="""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '<=')

    def __eq__(self, other) -> 'Filter':
        """Binary operator =="""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '==')

    def __ne__(self, other) -> 'Filter':
        """Binary operator !="""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '!=')

    def __gt__(self, other) -> 'Filter':
        """Binary operator >"""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '>')

    def __ge__(self, other) -> 'Filter':
        """Binary operator >="""
        from vegas.pipeline.filters import BinaryCompare
        return BinaryCompare(self, other, '>=')


class Factor(Term):
    """
    A Term that computes a numerical result for each asset on each day.
    
    Factors are the most common type of Pipeline expression, representing
    quantities like momentum, volatility, etc.
    """
    dtype = np.float64
    missing_value = np.nan

    def __add__(self, other) -> 'Factor':
        """Binary operator +"""
        return BinaryFactor(self, other, '+')
        
    def __sub__(self, other) -> 'Factor':
        """Binary operator -"""
        return BinaryFactor(self, other, '-')
        
    def __mul__(self, other) -> 'Factor':
        """Binary operator *"""
        return BinaryFactor(self, other, '*')
        
    def __truediv__(self, other) -> 'Factor':
        """Binary operator /"""
        return BinaryFactor(self, other, '/')

    def rank(self, method='ordinal', ascending=True, mask=None) -> 'Factor':
        """
        Construct a new Factor representing the sorted rank of each column within each row.
        
        Parameters
        ----------
        method : str, {'ordinal', 'min', 'max', 'dense', 'average'}, optional
            The method used to assign ranks to tied elements.
        ascending : bool, optional
            Whether to rank in ascending or descending order.
        mask : Filter, optional
            A Filter representing assets to consider when computing ranks.
            
        Returns
        -------
        Factor
            A new factor that will compute the ranking.
        """
        from vegas.pipeline.factors.rank import Rank
        return Rank(self, method=method, ascending=ascending, mask=mask or self.mask)
    
    def top(self, N, mask=None) -> 'Filter':
        """
        Construct a Filter matching the top N asset values of self each day.
        
        Parameters
        ----------
        N : int
            Number of assets passing the filter each day.
        mask : Filter, optional
            A Filter representing assets to consider when computing ranks.
            
        Returns
        -------
        Filter
            A filter matching the top N assets.
        """
        return self.rank(ascending=False, mask=mask).top(N)
    
    def bottom(self, N, mask=None) -> 'Filter':
        """
        Construct a Filter matching the bottom N asset values of self each day.
        
        Parameters
        ----------
        N : int
            Number of assets passing the filter each day.
        mask : Filter, optional
            A Filter representing assets to consider when computing ranks.
            
        Returns
        -------
        Filter
            A filter matching the bottom N assets.
        """
        return self.rank(ascending=True, mask=mask).top(N)
    
    def zscore(self, mask=None) -> 'Factor':
        """
        Construct a Factor that Z-Scores each day's results.
        
        Parameters
        ----------
        mask : Filter, optional
            A Filter defining values to include when computing Z-Scores.
            
        Returns
        -------
        Factor
            A Factor producing Z-scored values.
        """
        from vegas.pipeline.factors.statistical import ZScore
        return ZScore(self, mask=mask or self.mask)


class BinaryFactor(Factor):
    """
    A Factor that applies a binary operator to two inputs.
    """
    
    def __init__(self, left, right, op):
        """
        Parameters
        ----------
        left : Factor or scalar
            Left operand.
        right : Factor or scalar
            Right operand.
        op : str
            Binary operator to apply.
        """
        self.left = left
        self.right = right
        self.op = op
        inputs = []
        if isinstance(left, Term):
            inputs.append(left)
        if isinstance(right, Term):
            inputs.append(right)
        window_length = max([t.window_length for t in inputs]) if inputs else 1
        super().__init__(inputs=inputs, window_length=window_length)
        
    def compute(self, today, assets, out, *inputs):
        """Apply the binary operation to inputs."""
        op_map = {
            '+': np.add,
            '-': np.subtract,
            '*': np.multiply,
            '/': np.divide,
        }
        
        try:
            if len(inputs) == 0:
                # Both inputs are scalars
                if self.op == '+':
                    out[:] = float(self.left) + float(self.right)
                elif self.op == '-':
                    out[:] = float(self.left) - float(self.right)
                elif self.op == '*':
                    out[:] = float(self.left) * float(self.right)
                elif self.op == '/':
                    out[:] = float(self.left) / float(self.right)
            elif len(inputs) == 1:
                # One input is a scalar, one is a Term
                if self.left is inputs[0]:
                    left = inputs[0].astype(np.float64)
                    right = float(self.right)
                else:
                    left = float(self.left)
                    right = inputs[0].astype(np.float64)
                
                op_map[self.op](left, right, out=out)
            else:
                # Both inputs are Terms
                left = inputs[0].astype(np.float64)
                right = inputs[1].astype(np.float64)
                op_map[self.op](left, right, out=out)
        except Exception as e:
            # If operation fails, fill with NaN
            out[:] = np.nan


class Filter(Term):
    """
    A Term that computes a boolean result for each asset on each day.
    
    Filters are used to exclude or include assets based on various criteria.
    """
    dtype = np.bool_
    missing_value = False
    
    def __and__(self, other) -> 'Filter':
        """Binary operator &"""
        return BinaryFilter(self, other, '&')
        
    def __or__(self, other) -> 'Filter':
        """Binary operator |"""
        return BinaryFilter(self, other, '|')
        
    def __invert__(self) -> 'Filter':
        """Unary operator ~"""
        return UnaryFilter(self, '~')


class BinaryFilter(Filter):
    """
    A Filter that combines two filters with a binary operator.
    """
    
    def __init__(self, left, right, op):
        """
        Parameters
        ----------
        left : Filter
            Left operand.
        right : Filter
            Right operand.
        op : str
            Binary operator to apply ('&' or '|').
        """
        self.left = left
        self.right = right
        self.op = op
        inputs = []
        if isinstance(left, Term):
            inputs.append(left)
        if isinstance(right, Term):
            inputs.append(right)
        window_length = max([t.window_length for t in inputs]) if inputs else 1
        super().__init__(inputs=inputs, window_length=window_length)
        
    def compute(self, today, assets, out, *inputs):
        """Apply the binary operation to inputs."""
        if len(inputs) == 0:
            # Both inputs are scalars
            if self.op == '&':
                out[:] = self.left & self.right
            elif self.op == '|':
                out[:] = self.left | self.right
        elif len(inputs) == 1:
            # One input is a scalar, one is a Term
            if self.left is inputs[0]:
                if self.op == '&':
                    out[:] = inputs[0] & self.right
                elif self.op == '|':
                    out[:] = inputs[0] | self.right
            else:
                if self.op == '&':
                    out[:] = self.left & inputs[0]
                elif self.op == '|':
                    out[:] = self.left | inputs[0]
        else:
            # Both inputs are Terms
            if self.op == '&':
                out[:] = inputs[0] & inputs[1]
            elif self.op == '|':
                out[:] = inputs[0] | inputs[1]


class UnaryFilter(Filter):
    """
    A Filter that applies a unary operator to an input filter.
    """
    
    def __init__(self, input_filter, op):
        """
        Parameters
        ----------
        input_filter : Filter
            The filter to apply the unary operator to.
        op : str
            Unary operator to apply ('~').
        """
        self.input_filter = input_filter
        self.op = op
        super().__init__(inputs=[input_filter], window_length=input_filter.window_length)
        
    def compute(self, today, assets, out, in_filter):
        """Apply the unary operation to input."""
        if self.op == '~':
            out[:] = ~in_filter


class Classifier(Term):
    """
    A Term that groups assets into categories.
    
    Classifiers are used to group assets by shared characteristics like sector.
    """
    dtype = np.int64
    missing_value = -1 
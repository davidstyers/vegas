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
        from vegas.pipeline.factors import Rank
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
        # Use a dedicated TopN filter over the factor directly.
        from vegas.pipeline.filters.advanced import TopN
        return TopN(self, int(N), ascending=False, mask=mask or self.mask)
    
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
        from vegas.pipeline.filters.advanced import TopN
        return TopN(self, int(N), ascending=True, mask=mask or self.mask)
    
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
    # Use dtype=object so NumPy ufuncs don't coerce bools to float64 when combining masks.
    # We'll ensure boolean dtype at the engine boundary before applying the screen.
    dtype = np.dtype(object)
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
        """Apply the binary operation to inputs, producing a strict boolean mask."""
        def to_bool_array(val, length):
            # Convert scalars or arrays to strict boolean ndarray of expected length.
            if isinstance(val, (bool, np.bool_)):
                return np.full(length, bool(val), dtype=bool)
            arr = np.asarray(val)
            # If object dtype (from upstream), coerce via truthiness to bool elementwise where possible.
            if arr.dtype == object:
                # Convert None/np.nan to False, truthy to True
                coerced = np.zeros(arr.shape, dtype=bool)
                it = np.nditer(arr, flags=['refs_ok', 'multi_index'], op_flags=['readonly'])
                for x in it:
                    v = x.item()
                    coerced[it.multi_index] = bool(v) if isinstance(v, (bool, np.bool_, int, float)) else bool(v is True)
                arr = coerced
            else:
                arr = arr.astype(bool, copy=False)
            # Ensure 1-D of correct length
            arr = np.ravel(arr)
            if arr.shape[0] != length:
                # Broadcast scalar-like
                if arr.shape[0] == 1:
                    arr = np.full(length, bool(arr[0]), dtype=bool)
                else:
                    # On mismatch, fail-safe to all False
                    arr = np.zeros(length, dtype=bool)
            return arr

        n = len(assets)
        if len(inputs) == 0:
            left = to_bool_array(self.left, n)
            right = to_bool_array(self.right, n)
        elif len(inputs) == 1:
            if self.left is inputs[0]:
                left = to_bool_array(inputs[0], n)
                right = to_bool_array(self.right, n)
            else:
                left = to_bool_array(self.left, n)
                right = to_bool_array(inputs[0], n)
        else:
            left = to_bool_array(inputs[0], n)
            right = to_bool_array(inputs[1], n)

        if self.op == '&':
            np.bitwise_and(left, right, out=out, dtype=bool)
        elif self.op == '|':
            np.bitwise_or(left, right, out=out, dtype=bool)
        else:
            # Unknown op -> all False
            out[:] = False


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
        """Apply the unary operation to input, producing strict boolean mask."""
        arr = np.asarray(in_filter)
        if arr.dtype == object:
            arr = np.array([bool(x) for x in arr.ravel()], dtype=bool)
        else:
            arr = arr.astype(bool, copy=False).ravel()
        if arr.shape[0] != len(assets):
            if arr.shape[0] == 1:
                arr = np.full(len(assets), bool(arr[0]), dtype=bool)
            else:
                arr = np.zeros(len(assets), dtype=bool)
        if self.op == '~':
            np.logical_not(arr, out=out)
        else:
            out[:] = False


class Classifier(Term):
    """
    A Term that groups assets into categories.
    
    Classifiers are used to group assets by shared characteristics like sector.
    """
    dtype = np.int64
    missing_value = -1 
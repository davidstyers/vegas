"""Base classes for Pipeline Terms.

This module defines the core building blocks for computations in the pipeline system.
"""
from typing import List, Optional, Union, Any
import polars as pl


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
        
    def to_expression(self) -> pl.Expr:
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
        raise NotImplementedError("Term subclasses must implement to_expression")
    
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
        
    def to_expression(self) -> pl.Expr:
        """Apply the binary operation to inputs."""
        if isinstance(self.left, Term):
            left_expr = self.left.to_expression()
        else:
            left_expr = pl.lit(self.left)

        if isinstance(self.right, Term):
            right_expr = self.right.to_expression()
        else:
            right_expr = pl.lit(self.right)

        if self.op == '+':
            return left_expr + right_expr
        elif self.op == '-':
            return left_expr - right_expr
        elif self.op == '*':
            return left_expr * right_expr
        elif self.op == '/':
            return left_expr / right_expr
        else:
            raise ValueError(f"Unknown operator: {self.op}")


class Filter(Term):
    """
    A Term that computes a boolean result for each asset on each day.
    
    Filters are used to exclude or include assets based on various criteria.
    """
    
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
        
    def to_expression(self) -> pl.Expr:
        """Apply the binary operation to inputs, producing a strict boolean mask."""
        if isinstance(self.left, Term):
            left_expr = self.left.to_expression()
        else:
            left_expr = pl.lit(self.left)

        if isinstance(self.right, Term):
            right_expr = self.right.to_expression()
        else:
            right_expr = pl.lit(self.right)

        if self.op == '&':
            return left_expr & right_expr
        elif self.op == '|':
            return left_expr | right_expr
        else:
            raise ValueError(f"Unknown operator: {self.op}")


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
        
    def to_expression(self) -> pl.Expr:
        """Apply the unary operation to input, producing strict boolean mask."""
        if self.op == '~':
            return ~self.input_filter.to_expression()
        else:
            raise ValueError(f"Unknown operator: {self.op}")


class Classifier(Term):
    """
    A Term that groups assets into categories.
    
    Classifiers are used to group assets by shared characteristics like sector.
    """
    dtype = pl.Int64
    missing_value = -1
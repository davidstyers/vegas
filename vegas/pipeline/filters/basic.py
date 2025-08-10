"""Basic filters for the Vegas pipeline system.

This module defines common basic filters like StaticAssets and comparison operations.
"""
import polars as pl
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
        
    def to_expression(self) -> pl.Expr:
        """
        Determine which assets match the static list.
        """
        return pl.col('symbol').is_in(list(self.assets))


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
    
    def to_expression(self) -> pl.Expr:
        """
        Apply the comparison operation.
        """
        if isinstance(self.left, Term):
            left_expr = self.left.to_expression()
        else:
            left_expr = pl.lit(self.left)

        if isinstance(self.right, Term):
            right_expr = self.right.to_expression()
        else:
            right_expr = pl.lit(self.right)

        if self.op == '<':
            return left_expr < right_expr
        elif self.op == '<=':
            return left_expr <= right_expr
        elif self.op == '==':
            return left_expr == right_expr
        elif self.op == '!=':
            return left_expr != right_expr
        elif self.op == '>=':
            return left_expr >= right_expr
        elif self.op == '>':
            return left_expr > right_expr
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
    
    def to_expression(self) -> pl.Expr:
        """
        Identify values that are not NaN.
        """
        return self.term.to_expression().is_not_nan()
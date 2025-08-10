"""Base classes for Pipeline Terms.

This module defines the core building blocks for computations in the pipeline system.
"""
from typing import List, Optional, Union, Any
import polars as pl


class Term:
    """Base class for all pipeline computations.

    Terms form the nodes of expression trees compiled by the pipeline engine
    into Polars expressions. Subclasses implement `to_expression`.
    """
    def __init__(self,
                 inputs: Optional[List['Term']] = None,
                 window_length: int = 1,
                 mask: Optional['Filter'] = None):
        """Initialize a `Term` with optional inputs and mask.

        :param inputs: Input terms whose values are required to compute this term.
        :type inputs: Optional[list[Term]]
        :param window_length: Number of rows of historical data needed.
        :type window_length: int
        :param mask: Optional filter defining the computation universe.
        :type mask: Optional[Filter]
        :returns: None
        :rtype: None
        :Example:
            >>> t = Term(inputs=[], window_length=1)
        """
        self.inputs = inputs or []
        self.window_length = window_length
        self.mask = mask
        self.name = None
        
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing this term.

        Subclasses must override and build a valid `pl.Expr` that the engine can
        evaluate. The expression should rely on column references provided by the
        engine (e.g., 'symbol', 'timestamp', or input-specific columns).

        :returns: Polars expression computing term values.
        :rtype: pl.Expr
        :raises NotImplementedError: If the subclass does not implement this method.
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
    """Numeric term producing per-asset values for each date.

    Factors are used for quantities such as momentum, volatility, etc. They
    support arithmetic operators and helper constructors like `rank`, `top`,
    and `bottom` to compose new terms fluently.
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
        """Return a factor whose values are the cross-sectional ranks.

        :param method: Rank method ('ordinal', 'min', 'max', 'dense', 'average').
        :type method: str
        :param ascending: Whether ranks are ascending (smallest=1).
        :type ascending: bool
        :param mask: Optional mask to restrict the ranking universe.
        :type mask: Optional[Filter]
        :returns: New `Factor` computing ranks per date.
        :rtype: Factor
        :Example:
            >>> f.rank(method='dense', ascending=False)
        """
        from vegas.pipeline.factors import Rank
        return Rank(self, method=method, ascending=ascending, mask=mask or self.mask)
    
    def top(self, N, mask=None) -> 'Filter':
        """Return a filter matching the top-N assets by the factor each day.

        :param N: Number of assets to pass the filter each day.
        :type N: int
        :param mask: Optional mask to restrict the ranking universe.
        :type mask: Optional[Filter]
        :returns: Filter selecting the top N assets.
        :rtype: Filter
        :Example:
            >>> f.top(50)
        """
        # Use a dedicated TopN filter over the factor directly.
        from vegas.pipeline.filters.advanced import TopN
        return TopN(self, int(N), ascending=False, mask=mask or self.mask)
    
    def bottom(self, N, mask=None) -> 'Filter':
        """Return a filter matching the bottom-N assets by the factor each day.

        :param N: Number of assets to pass the filter each day.
        :type N: int
        :param mask: Optional mask to restrict the ranking universe.
        :type mask: Optional[Filter]
        :returns: Filter selecting the bottom N assets.
        :rtype: Filter
        :Example:
            >>> f.bottom(50)
        """
        from vegas.pipeline.filters.advanced import TopN
        return TopN(self, int(N), ascending=True, mask=mask or self.mask)
    
    def zscore(self, mask=None) -> 'Factor':
        """Return a factor that Z-scores each day's cross-section.

        :param mask: Optional mask defining values to include.
        :type mask: Optional[Filter]
        :returns: Factor producing standardized values.
        :rtype: Factor
        :Example:
            >>> f.zscore()
        """
        from vegas.pipeline.factors.statistical import ZScore
        return ZScore(self, mask=mask or self.mask)


class BinaryFactor(Factor):
    """Factor produced by applying a binary operator to two inputs."""
    
    def __init__(self, left, right, op):
        """Initialize a `BinaryFactor`.

        :param left: Left operand as `Term` or scalar.
        :type left: Term | Any
        :param right: Right operand as `Term` or scalar.
        :type right: Term | Any
        :param op: Binary operator, one of '+', '-', '*', '/'.
        :type op: str
        :raises ValueError: If an unknown operator is supplied.
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
        """Return the Polars expression applying the operator to inputs.

        :returns: Polars expression for the binary operation.
        :rtype: pl.Expr
        :raises ValueError: If an unknown operator is supplied.
        """
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
    """Boolean term used to include/exclude assets based on criteria."""
    
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
    """Filter combining two inputs with a binary boolean operator."""
    
    def __init__(self, left, right, op):
        """Initialize a `BinaryFilter`.

        :param left: Left filter operand.
        :type left: Filter
        :param right: Right filter operand.
        :type right: Filter
        :param op: Binary boolean operator ('&' or '|').
        :type op: str
        :raises ValueError: If an unknown operator is supplied.
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
        """Return a boolean Polars expression applying the binary operator.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        :raises ValueError: If an unknown operator is supplied.
        """
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
    """Filter applying a unary boolean operator to another filter."""
    
    def __init__(self, input_filter, op):
        """Initialize a `UnaryFilter`.

        :param input_filter: Input filter operand.
        :type input_filter: Filter
        :param op: Unary operator ('~').
        :type op: str
        :raises ValueError: If an unknown operator is supplied.
        """
        self.input_filter = input_filter
        self.op = op
        super().__init__(inputs=[input_filter], window_length=input_filter.window_length)
        
    def to_expression(self) -> pl.Expr:
        """Return a boolean expression applying the unary operator.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        :raises ValueError: If an unknown operator is supplied.
        """
        if self.op == '~':
            return ~self.input_filter.to_expression()
        else:
            raise ValueError(f"Unknown operator: {self.op}")


class Classifier(Term):
    """Term that groups assets into categories (e.g., sector identifiers)."""
    dtype = pl.Int64
    missing_value = -1
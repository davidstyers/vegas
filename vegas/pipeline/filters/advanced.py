"""Advanced filters for the Vegas pipeline system.

This module defines more complex filters like All, Any, and AtLeastN.
"""
import polars as pl
from vegas.pipeline.terms import Filter, Term


class All(Filter):
    """Filter that requires all inputs to be True.

    :param filters: Filters to combine with logical AND.
    :type filters: list[Filter]
    :Example:
        >>> mask = All(f1, f2, f3)
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression for logical AND over inputs.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        expr = self.filters[0].to_expression()
        for f in self.filters[1:]:
            expr = expr & f.to_expression()
        return expr


class Any(Filter):
    """Filter that requires at least one input to be True.

    :param filters: Filters to combine with logical OR.
    :type filters: list[Filter]
    :Example:
        >>> mask = Any(f1, f2)
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression for logical OR over inputs.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        expr = self.filters[0].to_expression()
        for f in self.filters[1:]:
            expr = expr | f.to_expression()
        return expr


class AtLeastN(Filter):
    """Filter requiring at least N of the inputs to be True.

    :param n: Minimum number of filters that must be True.
    :type n: int
    :param filters: Filters to evaluate.
    :type filters: list[Filter]
    :raises ValueError: If ``n`` is < 1 or greater than number of filters.
    :Example:
        >>> mask = AtLeastN(2, f1, f2, f3)
    """
    
    def __init__(self, n, *filters):
        self.n = n
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if n > len(filters):
            raise ValueError(f"n ({n}) cannot exceed the number of filters ({len(filters)})")
        
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def to_expression(self) -> pl.Expr:
        """Return a boolean expression for at least N True inputs.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        return pl.sum_horizontal([f.to_expression() for f in self.filters]) >= self.n


class NotMissing(Filter):
    """Filter selecting assets with non-missing data for a term.

    :param term: Term whose values are checked for missingness.
    :type term: Term
    :Example:
        >>> mask = NotMissing(my_factor)
    """
    
    def __init__(self, term):
        self.term = term
        super().__init__(inputs=[term], window_length=term.window_length)
    
    def to_expression(self) -> pl.Expr:
        """Return a boolean mask for non-missing values.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        return self.term.to_expression().is_not_null()


class TopN(Filter):
    """Filter selecting top-N assets by a factor's most recent value.

    :param term: Factor-like term providing numeric values.
    :type term: Term
    :param n: Number of assets to select.
    :type n: int
    :param ascending: If True, select smallest N; if False, select largest N.
    :type ascending: bool
    :raises ValueError: If ``n`` < 1.
    :Example:
        >>> mask = TopN(term=my_factor, n=100, ascending=False)
    """
    def __init__(self, term: Term, n: int, ascending: bool = False, mask: Filter | None = None):
        if n < 1:
            raise ValueError(f"TopN requires n>=1, got {n}")
        self.term = term
        self.n = int(n)
        self.ascending = bool(ascending)
        # Respect provided mask or term.mask by threading it into the Filter base via self.mask
        super().__init__(inputs=[term], window_length=term.window_length, mask=mask or getattr(term, 'mask', None))
    
    def to_expression(self) -> pl.Expr:
        """Return a boolean expression for top-N selection.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        return self.term.to_expression().rank(descending=not self.ascending) <= self.n
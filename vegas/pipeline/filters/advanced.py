"""Advanced filters for the Vegas pipeline system.

This module defines more complex filters like All, Any, and AtLeastN.
"""
import polars as pl
from vegas.pipeline.terms import Filter, Term


class All(Filter):
    """
    A Filter that requires all inputs to be True.
    
    Parameters
    ----------
    filters : list[Filter]
        The filters to combine with AND logic.
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def to_expression(self) -> pl.Expr:
        """
        Compute the logical AND of all input filters.
        """
        expr = self.filters[0].to_expression()
        for f in self.filters[1:]:
            expr = expr & f.to_expression()
        return expr


class Any(Filter):
    """
    A Filter that requires at least one input to be True.
    
    Parameters
    ----------
    filters : list[Filter]
        The filters to combine with OR logic.
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def to_expression(self) -> pl.Expr:
        """
        Compute the logical OR of all input filters.
        """
        expr = self.filters[0].to_expression()
        for f in self.filters[1:]:
            expr = expr | f.to_expression()
        return expr


class AtLeastN(Filter):
    """
    A Filter requiring at least N of the inputs filters to be True.
    
    Parameters
    ----------
    n : int
        Minimum number of filters that must be True.
    filters : list[Filter]
        The filters to check.
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
        """
        Compute whether at least N of the input filters are True.
        """
        return pl.sum_horizontal([f.to_expression() for f in self.filters]) >= self.n


class NotMissing(Filter):
    """
    A Filter selecting assets with non-missing data.
    
    Parameters
    ----------
    term : Term
        Term to check for missing values.
    """
    
    def __init__(self, term):
        self.term = term
        super().__init__(inputs=[term], window_length=term.window_length)
    
    def to_expression(self) -> pl.Expr:
        """
        Identify assets with non-missing data.
        """
        return self.term.to_expression().is_not_null()


class TopN(Filter):
    """
    A Filter selecting the top N assets by a factor's value on the most recent row.
    
    Parameters
    ----------
    term : Term
        Factor-like term providing numeric values.
    n : int
        Number of assets to select.
    ascending : bool
        If True, select smallest values; if False, select largest values.
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
        """
        Produce a boolean mask for top N selection on the most recent row.
        """
        return self.term.to_expression().rank(descending=not self.ascending) <= self.n
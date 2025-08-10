"""Statistical factors for the Vegas pipeline system.

This module defines factors for statistical transformations like Z-Score and ranking.
"""
import polars as pl
from vegas.pipeline.factors.custom import CustomFactor
from vegas.pipeline.terms import Term


class ZScore(CustomFactor):
    """
    Factor producing Z-scores for each day's cross-section.
    
    Z-scores are computed using:
    
    z = (x - mean(x)) / std(x)
    
    where x is the input data on a given day.
    
    Parameters
    ----------
    inputs : list, optional
        A list of data inputs to use in compute.
    window_length : int, optional
        The number of rows of data to pass to compute.
    mask : Filter, optional
        A Filter defining values to compute.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, mask=None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def to_expression(self) -> pl.Expr:
        """
        Compute Z-scores for the input data.
        """
        expr = self.term.to_expression()
        return (expr - expr.mean().over('date')) / (expr.std().over('date'))


class Rank(CustomFactor):
    """
    Factor representing the sorted rank of each column within each row.
    
    Parameters
    ----------
    term : Term
        The term to rank
    method : {'ordinal', 'min', 'max', 'dense', 'average'}, optional
        The method used to assign ranks to tied elements.
    ascending : bool, optional
        Whether to rank in ascending or descending order.
    mask : Filter, optional
        A Filter representing assets to consider when computing ranks.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, method='ordinal', ascending=True, mask=None):
        self.term = term
        self.method = method
        self.ascending = ascending
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def to_expression(self) -> pl.Expr:
        """
        Compute ranks for the input data.
        """
        return self.term.to_expression().rank(method=self.method, descending=not self.ascending).over('date')


class Percentile(CustomFactor):
    """
    Factor representing percentiles of data.
    
    Parameters
    ----------
    term : Term
        The term to compute percentiles for
    mask : Filter, optional
        A Filter representing assets to consider when computing percentiles.
    """
    window_length = 1  # Default to 1, since we only need the most recent values
    
    def __init__(self, term, mask=None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )
    
    def to_expression(self) -> pl.Expr:
        """
        Compute percentiles for the input data.
        """
        expr = self.term.to_expression()
        return expr.rank(method='ordinal').over('date') / expr.count().over('date')
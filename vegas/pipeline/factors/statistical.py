"""Statistical factors for the Vegas pipeline system.

This module defines factors for statistical transformations like Z-Score and ranking.
"""

from typing import Optional

import polars as pl

from vegas.pipeline.factors.custom import CustomFactor


class ZScore(CustomFactor):
    """Z-score the cross-section for each day.

    Z-scores are computed as z = (x - mean(x)) / std(x) within each date group.

    :param term: Input term whose values will be standardized.
    :type term: Term
    :param mask: Optional mask to restrict computation.
    :type mask: Optional[Filter]
    :Example:
        >>> z = ZScore(my_factor)
    """

    window_length = 1  # Default to 1, since we only need the most recent values

    def __init__(self, term, mask: Optional[object] = None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )

    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing Z-scores by date.

        :returns: Polars expression of standardized values.
        :rtype: pl.Expr
        """
        expr = self.term.to_expression()
        return (expr - expr.mean().over("date")) / (expr.std().over("date"))


class Rank(CustomFactor):
    """Rank values within each date cross-section.

    :param term: Term to rank.
    :type term: Term
    :param method: Rank method ('ordinal', 'min', 'max', 'dense', 'average').
    :type method: str
    :param ascending: Whether smaller values receive lower ranks.
    :type ascending: bool
    :param mask: Optional mask to restrict ranking universe.
    :type mask: Optional[Filter]
    :Example:
        >>> r = Rank(my_factor, method='dense', ascending=False)
    """

    window_length = 1  # Default to 1, since we only need the most recent values

    def __init__(
        self,
        term,
        method: str = "ordinal",
        ascending: bool = True,
        mask: Optional[object] = None,
    ):
        self.term = term
        self.method = method
        self.ascending = ascending
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )

    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing ranks by date.

        :returns: Polars expression of ranks.
        :rtype: pl.Expr
        """
        return (
            self.term.to_expression()
            .rank(method=self.method, descending=not self.ascending)
            .over("date")
        )


class Percentile(CustomFactor):
    """Percentile of a term within each date cross-section.

    :param term: Term to compute percentiles for.
    :type term: Term
    :param mask: Optional mask to restrict computation universe.
    :type mask: Optional[Filter]
    :Example:
        >>> p = Percentile(my_factor)
    """

    window_length = 1  # Default to 1, since we only need the most recent values

    def __init__(self, term, mask: Optional[object] = None):
        self.term = term
        super().__init__(
            inputs=[term],
            window_length=term.window_length,
            mask=mask or term.mask,
        )

    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing percentiles by date.

        :returns: Polars expression of percentiles in [0,1].
        :rtype: pl.Expr
        """
        expr = self.term.to_expression()
        return expr.rank(method="ordinal").over("date") / expr.count().over("date")

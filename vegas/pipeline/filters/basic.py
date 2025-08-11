"""Basic filters for the Vegas pipeline system.

This module defines common basic filters like StaticAssets and comparison operations.
"""

import polars as pl

from vegas.pipeline.terms import Filter, Term


class StaticAssets(Filter):
    """Filter for a static allow-list of asset symbols.

    :param assets: Symbols to include in the filter mask.
    :type assets: list[str]
    :Example:
        >>> mask = StaticAssets(["AAPL", "MSFT"])
    """

    def __init__(self, assets: list[str]):
        self.assets = set(assets)
        super().__init__()

    def to_expression(self) -> pl.Expr:
        """Return a boolean expression marking static assets.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        return pl.col("symbol").is_in(list(self.assets))


class BinaryCompare(Filter):
    """Filter representing a binary comparison between two values/terms.

    :param left: Left operand (`Term` or scalar).
    :type left: Term | Any
    :param right: Right operand (`Term` or scalar).
    :type right: Term | Any
    :param op: Comparison operator, one of '<', '<=', '==', '!=', '>=', '>'.
    :type op: str
    :raises ValueError: If an unknown operator is supplied.
    :Example:
        >>> mask = BinaryCompare(factor, 0, '>')
    """

    def __init__(self, left: Term | float | int, right: Term | float | int, op: str):
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
        """Return a boolean expression for the comparison.

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

        if self.op == "<":
            return left_expr < right_expr
        elif self.op == "<=":
            return left_expr <= right_expr
        elif self.op == "==":
            return left_expr == right_expr
        elif self.op == "!=":
            return left_expr != right_expr
        elif self.op == ">=":
            return left_expr >= right_expr
        elif self.op == ">":
            return left_expr > right_expr
        else:
            raise ValueError(f"Unknown comparison operator: {self.op}")


class NotNaN(Filter):
    """Filter returning True where a term is not NaN.

    :param term: Term whose values are tested.
    :type term: Term
    :Example:
        >>> mask = NotNaN(my_factor)
    """

    def __init__(self, term):
        self.term = term
        super().__init__(inputs=[term], window_length=term.window_length)

    def to_expression(self) -> pl.Expr:
        """Return a boolean expression for non-NaN values.

        :returns: Polars boolean expression.
        :rtype: pl.Expr
        """
        return self.term.to_expression().is_not_nan()

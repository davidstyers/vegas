"""Custom factor implementation for the Vegas pipeline system.

This module defines the CustomFactor class that allows users to create their own factors.
"""

from typing import Any, ClassVar, Optional, Sequence

import polars as pl

from vegas.pipeline.terms import Factor


class CustomFactor(Factor):
    """Base class for user-defined numeric Factors.

    Subclasses must implement `to_expression` to return a Polars expression.

    Example:
        >>> class SimpleMovingAverage(CustomFactor):
        ...     inputs = ['close']
        ...     window_length = 10
        ...     def to_expression(self):
        ...         return pl.col(self.inputs[0]).rolling_mean(self.window_length).over('symbol')
        ...
        >>> sma_10 = SimpleMovingAverage()

    """

    # Class-level defaults that strategy authors override in subclasses.
    # These are class variables on purpose and do not conflict with Term instance attributes.
    inputs: ClassVar[Optional[Sequence[str]]] = None
    window_length: ClassVar[Optional[int]] = None

    def __init__(
        self,
        inputs: Optional[Sequence[str]] = None,
        window_length: Optional[int] = None,
        mask: Optional[object] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `CustomFactor`.

        :param inputs: Input column names or `Term` objects.
        :type inputs: list | None
        :param window_length: Number of rows included in the window.
        :type window_length: int | None
        :param mask: Optional mask defining the computation universe.
        :type mask: Optional[vegas.pipeline.terms.Filter]
        :param kwargs: Additional attributes to set on the instance.
        :type kwargs: dict
        :raises ValueError: If subclass fails to supply required `inputs` or `window_length`.
        :returns: None
        :rtype: None
        :Example:
            >>> cf = CustomFactor(inputs=['close'], window_length=5)
        """
        if inputs is None:
            if self.inputs is None:
                raise ValueError(
                    f"{type(self).__name__} requires 'inputs' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            inputs = self.inputs

        if window_length is None:
            if self.window_length is None:
                raise ValueError(
                    f"{type(self).__name__} requires 'window_length' to be specified as a "
                    "class attribute or parameter to __init__"
                )
            window_length = self.window_length

        # Pass through to Term with the resolved values.
        super().__init__(inputs=inputs, window_length=int(window_length), mask=mask)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_expression(self) -> pl.Expr:
        """Return the Polars expression implementing the factor.

        :returns: Polars expression to be evaluated by the pipeline engine.
        :rtype: pl.Expr
        :raises NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "CustomFactor subclasses must implement to_expression"
        )

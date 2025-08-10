"""Basic built-in factors for the Vegas pipeline system.

This module defines common basic factors like Returns and moving averages.
"""
import polars as pl
from vegas.pipeline.factors.custom import CustomFactor


class Returns(CustomFactor):
    """Percentage change in price over a lookback window.

    :param inputs: Column names to use as inputs (defaults to ['close']).
    :type inputs: list[str]
    :param window_length: Number of rows to compute the change over (>=2 for pct_change).
    :type window_length: int
    :returns: Polars expression computing percentage change per symbol.
    :rtype: pl.Expr
    :Example:
        >>> Returns(inputs=["close"], window_length=2)
    """
    inputs = ['close']
    window_length = 2  # Default to daily returns
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression for computing returns.

        :returns: Polars expression computing percentage change by symbol.
        :rtype: pl.Expr
        :Example:
            >>> expr = Returns().to_expression()
        """
        return pl.col(self.inputs[0]).pct_change(n=self.window_length - 1).over('symbol')


class SimpleMovingAverage(CustomFactor):
    """Simple moving average of a data input.

    Example:
        >>> sma = SimpleMovingAverage(inputs=['close'], window_length=10)
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day moving average
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression for SMA.

        :returns: Polars expression computing SMA by symbol.
        :rtype: pl.Expr
        :Example:
            >>> expr = SimpleMovingAverage().to_expression()
        """
        return pl.col(self.inputs[0]).rolling_mean(self.window_length).over('symbol')


class ExponentialWeightedMovingAverage(CustomFactor):
    """Exponentially-weighted moving average of a data input.

    :param inputs: Input column(s) (default ['close']).
    :type inputs: list[str] | None
    :param window_length: Number of rows included in the window.
    :type window_length: int | None
    :param decay_rate: Weight decay factor (higher emphasizes recent values).
    :type decay_rate: float
    :Example:
        >>> ewma = ExponentialWeightedMovingAverage(decay_rate=0.3)
    """
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day moving average
    
    def __init__(self, inputs=None, window_length=None, decay_rate=0.5, mask=None):
        self.decay_rate = decay_rate
        super().__init__(inputs=inputs, window_length=window_length, mask=mask)
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression for EWMA.

        :returns: Polars expression computing EWMA by symbol.
        :rtype: pl.Expr
        :Example:
            >>> expr = ExponentialWeightedMovingAverage().to_expression()
        """
        return pl.col(self.inputs[0]).ewm_mean(alpha=self.decay_rate, adjust=False).over('symbol')


class VWAP(CustomFactor):
    """Volume Weighted Average Price over the window.

    VWAP is the ratio of traded value to traded volume over a window (often one day).
    """
    inputs = ['close', 'volume']
    window_length = 1  # Default to single day VWAP
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing VWAP.

        :returns: Polars expression for VWAP by symbol.
        :rtype: pl.Expr
        :Example:
            >>> expr = VWAP().to_expression()
        """
        return (pl.col('close') * pl.col('volume')).sum().over('symbol') / pl.col('volume').sum().over('symbol')


class StandardDeviation(CustomFactor):
    """Rolling standard deviation of a data input over a window."""
    inputs = ['close']  # Default to close prices
    window_length = 10  # Default to 10-day window
    
    def to_expression(self) -> pl.Expr:
        """Return the Polars expression computing rolling std by symbol.

        :returns: Polars expression computing standard deviation.
        :rtype: pl.Expr
        :Example:
            >>> expr = StandardDeviation().to_expression()
        """
        return pl.col(self.inputs[0]).rolling_std(self.window_length).over('symbol')
"""Pipeline computation engine for Vegas.

This module implements the `PipelineEngine`, which evaluates `Pipeline`
definitions against historical market data provided by the `DataPortal`.
The engine is designed to be deterministic and side-effect free: it builds
Polars expressions from `Term` objects and executes them over sanitized input
data. Pipelines are typically computed once per simulation day by the
`BacktestEngine` and cached for strategy access via `pipeline_output(name)`.
"""

import logging
from datetime import date, datetime
from typing import Union

import polars as pl
import pytz

from vegas.pipeline.pipeline import Pipeline


class PipelineEngine:
    """Engine responsible for evaluating `Pipeline` objects.

    The engine compiles `Term` trees into Polars expressions and evaluates them
    over the minimal lookback window required by the pipeline. Inputs are
    sanitized for symbol dtype consistency and missing values to keep pipeline
    results predictable and easier to consume.

    Example:
        >>> from vegas.pipeline.pipeline import Pipeline
        >>> from vegas.pipeline.factors.basic import SimpleMovingAverage
        >>> engine = PipelineEngine(data_portal)
        >>> pipe = Pipeline(columns={"sma10": SimpleMovingAverage(inputs=["close"], window_length=10)})
        >>> result = engine.run_pipeline(pipe, start_date, end_date)

    """

    def __init__(self, data_portal) -> None:
        """Initialize a new engine bound to a `DataPortal`.

        :param data_portal: Data access layer used to fetch history windows.
        :type data_portal: vegas.data.DataPortal
        :raises Exception: Propagates if the supplied `data_portal` is misconfigured.
        :returns: None
        :rtype: None
        :Example:
            >>> engine = PipelineEngine(data_portal)
        """
        self.data_portal = data_portal
        self.logger = logging.getLogger("vegas.pipeline.engine")

    def run_pipeline(
        self,
        pipeline: Pipeline,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pl.DataFrame:
        """Compute a pipeline over a date range and return results.

        The engine determines the maximal lookback required by the pipeline,
        fetches a history window from the `DataPortal` ending at the engine's
        current datetime, and evaluates the pipeline expressions. If a screen
        is present, the final results are filtered accordingly.

        :param pipeline: Pipeline definition containing columns and optional screen.
        :type pipeline: vegas.pipeline.pipeline.Pipeline
        :param start_date: Inclusive start date as `datetime` or ISO-like string.
        :type start_date: Union[str, datetime]
        :param end_date: Inclusive end date as `datetime` or ISO-like string.
        :type end_date: Union[str, datetime]
        :returns: Polars DataFrame with columns `[timestamp, symbol, *pipeline_columns]`.
        :rtype: polars.DataFrame
        :raises Exception: If inputs cannot be parsed or the data layer fails to provide data.
        :Example:
            >>> df = engine.run_pipeline(pipe, "2023-01-01", "2023-01-31")
        """
        start_date = self._to_datetime(start_date, tz=self.data_portal.timezone)
        end_date = self._to_datetime(end_date, tz=self.data_portal.timezone)

        max_window = self._get_max_window_length(pipeline)

        # Retrieve market data for the lookback window up to current_dt explicitly
        end_dt = getattr(self.data_portal, "_current_dt", None)
        market_data = self.data_portal.history(
            assets=None,  # All assets
            fields=None,  # All fields
            bar_count=max_window,
            frequency=pipeline.frequency,
            end_dt=end_dt,
        )

        # Sanitize symbols early to prevent None/invalid entries from propagating
        try:
            if not market_data.is_empty():
                if "symbol" in market_data.columns:
                    if market_data.get_column("symbol").dtype != pl.Utf8:
                        market_data = market_data.with_columns(
                            pl.col("symbol").cast(pl.Utf8)
                        )
                    market_data = market_data.filter(
                        pl.col("symbol").is_not_null() & (pl.col("symbol") != "")
                    )
        except Exception:
            pass

        if market_data.is_empty():
            self.logger.warning(
                f"No market data found between {start_date.date()} and {end_date.date()}"
            )
            return pl.DataFrame()

        # Build the expressions for all terms
        expressions = [
            term.to_expression().alias(name) for name, term in pipeline.columns.items()
        ]

        # Add the screen expression if it exists
        if pipeline.screen:
            expressions.append(pipeline.screen.to_expression().alias("_screen_mask"))

        # Run the expressions over the market data
        result = (
            market_data.lazy().with_columns(expressions).collect(engine="streaming")
        )

        # Final sanitization of result symbols
        try:
            if not result.is_empty() and "symbol" in result.columns:
                if result.get_column("symbol").dtype != pl.Utf8:
                    result = result.with_columns(pl.col("symbol").cast(pl.Utf8))
                result = result.filter(
                    pl.col("symbol").is_not_null() & (pl.col("symbol") != "")
                )
        except Exception:
            pass

        # Apply the screen if it exists
        if pipeline.screen:
            result = result.filter(pl.col("_screen_mask")).drop("_screen_mask")

        # Keep explicit columns; put date and symbol first
        ordered_cols = ["timestamp", "symbol"] + list(pipeline.columns.keys())
        result = result.select(ordered_cols)

        return result

    def _get_max_window_length(self, pipeline: Pipeline) -> int:
        """Return the maximum lookback window length required by a pipeline.

        :param pipeline: Pipeline to analyze for window requirements.
        :type pipeline: vegas.pipeline.pipeline.Pipeline
        :returns: The maximum window length across all columns and screen.
        :rtype: int
        :Example:
            >>> max_window = engine._get_max_window_length(pipe)
        """
        max_window = 1  # Default minimum

        # Check each column
        for term in pipeline.columns.values():
            max_window = max(max_window, term.window_length)

        # Check screen
        if pipeline.screen is not None:
            max_window = max(max_window, pipeline.screen.window_length)

        return max_window

    def _to_datetime(self, value: Union[str, datetime, date], tz: str) -> datetime:
        """Convert supported inputs to a timezone-aware ``datetime``.

        :param value: Input as `datetime`, `date`, or ISO-like string.
        :type value: Union[str, datetime, date]
        :param tz: Target timezone name to apply.
        :type tz: str
        :returns: Timezone-aware datetime in the requested timezone.
        :rtype: datetime
        :raises TypeError: If the input type is unsupported.
        :raises ValueError: If string parsing fails.
        :Example:
            >>> engine._to_datetime("2024-01-01", tz="UTC")
        """
        if isinstance(value, datetime):
            return value.astimezone(pytz.timezone(tz))
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time()).astimezone(
                pytz.timezone(tz)
            )
        if isinstance(value, str):
            # Try multiple common formats; fallback to fromisoformat
            for fmt in (
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d",
                "%Y/%m/%d %H:%M:%S",
            ):
                try:
                    from datetime import datetime as _dt

                    return _dt.strptime(value, fmt)
                except Exception:
                    pass
            # Last resort
            try:
                from datetime import datetime as _dt

                return _dt.fromisoformat(value)
            except Exception:
                self.logger.error(f"Unable to parse date string: {value}")
                raise
        raise TypeError(f"Unsupported date type: {type(value)}")

"""Pipeline class implementation for the Vegas backtesting engine.

This module defines the Pipeline class that represents a collection of computations
to be executed at the start of each trading day.
"""

from typing import Any, Dict, Optional


class Pipeline:
    """Container for computations evaluated once per trading day.

    A `Pipeline` gathers columns defined by `Term` objects (Factors, Filters,
    Classifiers) and an optional screen. The engine evaluates the pipeline over
    historical data and returns a Polars DataFrame for the given date(s).
    """

    def __init__(
        self,
        columns: Optional[Dict[str, Any]] = None,
        screen: Any | None = None,
        data_portal: Any | None = None,
        frequency: str = "1h",
    ) -> None:
        """Initialize a pipeline definition.

        :param columns: Mapping from result column name to `Term` (Factor/Filter/Classifier).
        :type columns: Optional[Dict[str, Any]]
        :param screen: Optional `Filter` to restrict the output universe.
        :type screen: Optional[vegas.pipeline.terms.Filter]
        :param data_portal: Optional `DataPortal` reference (not required for evaluation).
        :type data_portal: Optional[Any]
        :param frequency: Frequency string used for historical data (e.g., '1h', '1d').
        :type frequency: str
        :returns: None
        :rtype: None
        :Example:
            >>> from vegas.pipeline import Pipeline
            >>> pipe = Pipeline(columns={"ret": my_factor}, screen=my_filter, frequency='1h')
        """
        self.columns = columns or {}
        self.screen = screen
        self.data_portal = data_portal
        self.frequency = frequency

    def add(self, term: Any, name: str, overwrite: bool = False) -> "Pipeline":
        """Add a `Term` to the pipeline.

        :param term: Term to add.
        :type term: vegas.pipeline.terms.Term
        :param name: Column name to assign to the computed term.
        :type name: str
        :param overwrite: If ``True``, replace an existing column with the same name.
        :type overwrite: bool
        :returns: Self, to allow fluent chaining.
        :rtype: Pipeline
        :raises KeyError: If the column already exists and ``overwrite`` is ``False``.
        :Example:
            >>> pipe.add(my_factor, name="momentum")
        """
        if name in self.columns and not overwrite:
            raise KeyError(
                f"Column '{name}' already exists. To overwrite, set overwrite=True."
            )
        self.columns[name] = term
        return self

    def remove(self, name: str) -> Any:
        """Remove a column from the pipeline.

        :param name: Name of the column to remove.
        :type name: str
        :returns: The removed `Term`.
        :rtype: vegas.pipeline.terms.Term
        :raises KeyError: If no such column exists.
        :Example:
            >>> removed = pipe.remove("momentum")
        """
        if name not in self.columns:
            raise KeyError(f"No column named '{name}' exists.")
        return self.columns.pop(name)

    def set_screen(self, screen: Any, overwrite: bool = False) -> "Pipeline":
        """Attach or replace the pipeline screen.

        :param screen: Filter used to restrict output rows.
        :type screen: vegas.pipeline.terms.Filter
        :param overwrite: If ``True``, replace any existing screen.
        :type overwrite: bool
        :returns: Self, to allow fluent chaining.
        :rtype: Pipeline
        :raises ValueError: If a screen already exists and ``overwrite`` is ``False``.
        :Example:
            >>> pipe.set_screen(my_filter, overwrite=True)
        """
        if self.screen is not None and not overwrite:
            raise ValueError(
                "Pipeline already has a screen. To overwrite, set overwrite=True."
            )
        self.screen = screen
        return self

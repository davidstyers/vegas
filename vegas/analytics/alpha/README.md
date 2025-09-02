# Alpha Submodule Design

This document outlines the file structure, API, and responsibilities for the `vegas.analytics.alpha` submodule.

## 1. File Structure

The `alpha` submodule will be organized as follows:

```
vegas/analytics/alpha/
├── __init__.py
├── engine.py
└── report.py
```

## 2. Component Responsibilities

### `__init__.py`
- **Purpose**: Makes the `alpha` submodule a Python package and exposes the public API.
- **Key Contents**:
  - `from .engine import Alpha`

### `engine.py`
- **Purpose**: Contains the core `Alpha` class that orchestrates the strategy triage process.
- **Key Classes/Functions**:
  - `class Alpha`: The main entry point for running the triage. It will handle parameter space iteration, data slicing, and coordinating the execution of strategies. It will instantiate and run the `BacktestEngine` for each parameter combination and filter the results using a user-provided decision function.

### `report.py`
- **Purpose**: Handles the generation of reports summarizing the triage results, utilizing `quantstats`.
- **Key Classes/Functions**:
  - `generate_report(results: Results, output_path: str)`: A function that takes the `Results` object from a backtest run and generates a `quantstats` report, saving it to the specified `output_path`.

## 3. Public API for `Alpha` Class

The public API for the `Alpha` class in `engine.py` will be as follows:

```python
from typing import Type, Dict, List, Tuple, Callable, Optional
from vegas.analytics import Results
from vegas.strategy import Strategy

class Alpha:
    def __init__(
        self,
        strategy: Type[Strategy],
        param_space: Dict[str, List],
        tickers: List[str],
        slices: List[Tuple[str, str]],
        horizon: int,
        decision_func: Optional[Callable[[Results], bool]] = None
    ):
        """
        Initializes the Alpha triage engine.

        Args:
            strategy: The strategy class to be evaluated.
            param_space: A dictionary defining the hyperparameter space to search.
            tickers: A list of ticker symbols to run the strategy on.
            slices: A list of (start_date, end_date) tuples for data slicing.
            horizon: The forward-looking period for evaluating strategy signals.
            decision_func: A function that takes a Results object and returns True if the strategy is promising.
        """
        ...

    def run(self) -> List[Results]:
        """
        Runs the strategy triage across the defined parameter space and data slices.

        Returns:
            A list of Results objects for the strategies that passed the decision function.
        """
        ...

    def report(self, results: List[Results], output_path: str):
        """
        Generates a quantstats report of the triage results.

        Args:
            results: A list of Results objects from the `run` method.
            output_path: The file path to save the report to (e.g., "report.html").
        """
        ...
"""Core engine for the Alpha submodule."""

from typing import Type, Dict, List, Tuple, Callable, Optional
from vegas.analytics import Results
from vegas.strategy import Strategy
from vegas.engine import BacktestEngine
from datetime import datetime
import itertools
import os
import polars as pl
import pandas as pd
from vegas.analytics.alpha import metrics, report, decision


class Alpha:
    """Strategy triage engine."""

    def __init__(self, signals: pl.DataFrame, prices: pl.DataFrame):
        self.signals = signals
        self.prices = prices

    def run(self) -> dict:
        """Runs the strategy triage across the defined parameter space and data slices."""
        all_results = {}
        param_combinations = list(itertools.product(*self.param_space.values()))

        for params in param_combinations:
            param_dict = dict(zip(self.param_space.keys(), params))
            
            all_preds = []
            all_actuals = []
            ic_over_time = []

            for start_str, end_str in self.slices:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = datetime.strptime(end_str, "%Y-%m-%d")

                engine = BacktestEngine()
                strategy_instance = self.strategy(**param_dict)
                strategy_instance = self.strategy(**param_dict)

                results = engine.run(
                    start=start,
                    end=end,
                    strategy=strategy_instance,
                )
                
                preds = results.positions["signal"]
                actuals = results.positions["returns"]
                
                all_preds.append(preds)
                all_actuals.append(actuals)
                ic_over_time.append(metrics.rank_information_coefficient(preds, actuals))

            # Aggregate results and calculate metrics
            aggregated_preds = pl.concat(all_preds)
            aggregated_actuals = pl.concat(all_actuals)

            calculated_metrics = {
                "rank_ic": metrics.rank_information_coefficient(aggregated_preds, aggregated_actuals),
                "ic_stability": metrics.ic_stability(pl.Series(ic_over_time)),
                "hit_rate": metrics.hit_rate(aggregated_preds, aggregated_actuals),
                "cost_adjusted_returns": metrics.cost_adjusted_returns(
                    aggregated_preds, aggregated_actuals, costs=0.001
                ),
            }
            calculated_metrics["viability"] = decision.apply_viability_rules(
                calculated_metrics
            )
            all_results[str(param_dict)] = calculated_metrics

        return all_results

    def report(self, results: dict, output_dir: str):
        """Generates a JSON report of the alpha analysis.

        Args:
            results: The results dictionary from the `run` method.
            output_dir: The directory to save the report file to.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "alpha_report.json")
        with open(output_path, "w") as f:
            f.write(report.generate_report(results))

    def forward_returns(self, horizons=[1, 5, 20]) -> dict[int, pd.DataFrame]:
        """Compute forward returns per horizon, aligned with signals."""
        # TODO: Implement forward returns calculation
        raise NotImplementedError

    def evaluate(self, horizons=[1, 5, 20]) -> pd.DataFrame:
        """
        Compute predictive metrics:
          - Spearman IC (per horizon, per asset)
          - Hit Rate (% sign agreement)
        Aggregate metrics across assets/time.
        """
        # TODO: Implement evaluation logic
        raise NotImplementedError

import json
import polars as pl


def generate_report(results: dict) -> str:
    """
    Generate a JSON report summarizing the metrics for each parameter configuration.
    """
    report = {}
    for params, metrics in results.items():
        param_str = json.dumps(params)
        report[param_str] = {
            "rank_ic": metrics.get("rank_ic"),
            "ic_stability": metrics.get("ic_stability"),
            "hit_rate": metrics.get("hit_rate"),
            "cost_adjusted_returns": metrics.get("cost_adjusted_returns"),
        }
    return json.dumps(report, indent=4)
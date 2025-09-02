import polars as pl
from scipy.stats import spearmanr


def rank_information_coefficient(preds: pl.Series, actuals: pl.Series) -> float:
    """
    Calculate the Rank Information Coefficient (Spearman's rank correlation).
    """
    return spearmanr(preds.rank(), actuals.rank()).correlation


def ic_stability(ic_over_time: pl.Series) -> dict:
    """
    Calculate IC stability metrics.
    """
    return {
        "ic_mean": ic_over_time.mean(),
        "ic_std": ic_over_time.std(),
        "ic_sign_consistency": (ic_over_time > 0).mean(),
    }


def hit_rate(preds: pl.Series, actuals: pl.Series) -> float:
    """
    Calculate the hit rate (% of correct direction).
    """
    return (pl.sign(preds) == pl.sign(actuals)).mean()


def cost_adjusted_returns(
    preds: pl.Series, actuals: pl.Series, costs: float
) -> float:
    """
    Calculate cost-adjusted rough returns.
    """
    return (preds.abs() - costs).mean()
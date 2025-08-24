"""
Alpha Module for Signal Evaluation

This module provides the Alpha class for evaluating predictive power 
of strategy-generated signals.
"""

from typing import Dict, List

import polars as pl
import numpy as np
from scipy.stats import spearmanr


class Alpha:
    """
    Alpha class for signal evaluation.
    
    Evaluates the predictive power of trading signals by computing forward returns
    and analyzing the correlation between signals and future price movements.
    """

    def __init__(self, signals: pl.DataFrame, prices: pl.DataFrame):
        """
        Initialize Alpha evaluator with signals and prices.

        :param signals: Polars DataFrame [datetime × assets] of strategy outputs
        :type signals: pl.DataFrame
        :param prices: Polars DataFrame [datetime × assets] of close prices
        :type prices: pl.DataFrame
        """
        self.signals = signals
        self.prices = prices
        
        # Validate inputs
        if "datetime" not in signals.columns:
            raise ValueError("Signals DataFrame must have a 'datetime' column")
        if "datetime" not in prices.columns:
            raise ValueError("Prices DataFrame must have a 'datetime' column")
        
        # Get common asset columns (exclude datetime)
        signal_assets = set(signals.columns) - {"datetime"}
        price_assets = set(prices.columns) - {"datetime"}
        self.common_assets = sorted(signal_assets.intersection(price_assets))
        
        if not self.common_assets:
            raise ValueError("No common assets found between signals and prices")

    def forward_returns(self, horizons: List[int] = [1, 5, 20]) -> Dict[int, pl.DataFrame]:
        """
        Compute forward returns for each horizon.

        For each horizon h, compute:
        fwd_ret[t, asset] = prices[t+h, asset] / prices[t, asset] - 1

        :param horizons: List of forward-looking horizons (in periods)
        :type horizons: List[int]
        :returns: Dict mapping each horizon to a Polars DataFrame [datetime × assets]
        :rtype: Dict[int, pl.DataFrame]
        """
        forward_returns = {}
        
        # Sort prices by datetime to ensure correct ordering
        sorted_prices = self.prices.sort("datetime")
        
        for horizon in horizons:
            # Create a DataFrame to collect forward returns for this horizon
            fwd_ret_data = {"datetime": sorted_prices["datetime"].to_list()}
            
            for asset in self.common_assets:
                if asset in sorted_prices.columns:
                    # Get the price series for this asset
                    prices_series = sorted_prices[asset].to_list()
                    
                    # Compute forward returns
                    fwd_returns = []
                    for i in range(len(prices_series)):
                        if i + horizon < len(prices_series):
                            current_price = prices_series[i]
                            future_price = prices_series[i + horizon]
                            
                            # Only compute return if both prices are not None
                            if current_price is not None and future_price is not None and current_price != 0:
                                fwd_return = (future_price / current_price) - 1.0
                                fwd_returns.append(fwd_return)
                            else:
                                fwd_returns.append(None)
                        else:
                            # Cannot compute forward return for the last 'horizon' periods
                            fwd_returns.append(None)
                    
                    fwd_ret_data[asset] = fwd_returns
                else:
                    # Asset not in prices, fill with None
                    fwd_ret_data[asset] = [None] * len(sorted_prices)
            
            forward_returns[horizon] = pl.DataFrame(fwd_ret_data)
        
        return forward_returns

    def evaluate(self, horizons: List[int] = [1, 5, 20]) -> pl.DataFrame:
        """
        Evaluate predictive power of signals.

        Metrics:
          - Spearman IC (per horizon, averaged across assets)
          - Hit Rate (% sign agreement between signal and forward returns)

        :param horizons: List of forward-looking horizons to evaluate
        :type horizons: List[int]
        :returns: Polars DataFrame with rows=horizons, cols=[IC, HitRate]
        :rtype: pl.DataFrame
        """
        # Compute forward returns for all horizons
        fwd_returns_dict = self.forward_returns(horizons)
        
        results_data = []
        
        for horizon in horizons:
            fwd_returns_df = fwd_returns_dict[horizon]
            
            # Align signals and forward returns DataFrames on datetime
            merged_df = self.signals.join(
                fwd_returns_df.rename({col: f"{col}_fwd" for col in fwd_returns_df.columns if col != "datetime"}),
                on="datetime",
                how="inner"
            )
            
            ic_values = []
            hit_rates = []
            
            # Compute metrics for each asset
            for asset in self.common_assets:
                signal_col = asset
                fwd_ret_col = f"{asset}_fwd"
                
                if signal_col in merged_df.columns and fwd_ret_col in merged_df.columns:
                    # Get valid (non-null) pairs of signals and forward returns
                    valid_data = merged_df.select([signal_col, fwd_ret_col]).filter(
                        (pl.col(signal_col).is_not_null()) & (pl.col(fwd_ret_col).is_not_null())
                    )
                    
                    if valid_data.height > 1:  # Need at least 2 points for correlation
                        signals_array = valid_data[signal_col].to_numpy()
                        returns_array = valid_data[fwd_ret_col].to_numpy()
                        
                        # Compute Spearman correlation (Information Coefficient)
                        try:
                            ic, _ = spearmanr(signals_array, returns_array)
                            if not np.isnan(ic):
                                ic_values.append(ic)
                        except:
                            pass  # Skip this asset if correlation computation fails
                        
                        # Compute Hit Rate (sign agreement)
                        signal_signs = np.sign(signals_array)
                        return_signs = np.sign(returns_array)
                        hit_rate = np.mean(signal_signs == return_signs)
                        if not np.isnan(hit_rate):
                            hit_rates.append(hit_rate)
            
            # Aggregate metrics across assets
            mean_ic = np.mean(ic_values) if ic_values else 0.0
            mean_hit_rate = np.mean(hit_rates) if hit_rates else 0.5
            
            results_data.append({
                "horizon": horizon,
                "IC": mean_ic,
                "HitRate": mean_hit_rate
            })
        
        return pl.DataFrame(results_data)

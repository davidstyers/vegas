"""Analytics and statistics layer for the Vegas backtesting engine.

This module provides functionality for analyzing backtest results,
generating statistics, and creating visualizations using QuantStats.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl

# Import quantstats for performance analytics
try:
    import quantstats as qs

    HAS_QS = True
except ImportError:
    HAS_QS = False
    logging.warning(
        "QuantStats not installed. Some analytics features will be unavailable."
    )


@dataclass
class Results:
    """Container for backtest results.

    Attributes:
        equity_curve: DataFrame with equity curve data
        trades: List of executed transactions
        stats: Dictionary of performance statistics
        positions_history: Historical positions data
        returns: DataFrame with return data
        benchmark_data: Optional benchmark comparison data

    """

    equity_curve: pl.DataFrame
    trades: List[Any]
    stats: Dict[str, Any]
    positions_history: Dict[datetime, Dict[str, Dict[str, float]]]
    returns: pl.DataFrame
    benchmark_data: Optional[pl.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for serialization.

        Returns:
            Dictionary representation of results

        """
        result = {
            "equity_curve": self.equity_curve.to_dicts(),
            "trades": self.trades,
            "stats": self.stats,
            "returns": self.returns.to_dicts(),
        }

        if self.benchmark_data is not None:
            result["benchmark_data"] = self.benchmark_data.to_dicts()

        return result

    def create_tearsheet(
        self,
        title: str = "Strategy Performance",
        benchmark_symbol: str = None,
        output_file: str = None,
        output_format: str = "html",
    ) -> None:
        """Create a comprehensive performance tearsheet using QuantStats.

        Args:
            title: Title of the tearsheet
            benchmark_symbol: Ticker symbol for benchmark (e.g. 'SPY')
            output_file: Path to save the tearsheet
            output_format: Format to save the tearsheet ('html' or 'pdf')

        Returns:
            None

        """
        if not HAS_QS:
            logging.error("QuantStats not installed. Cannot create tearsheet.")
            return

        # Prepare returns series - convert polars DataFrame to pandas Series for QuantStats
        returns_pd = self.returns.to_pandas()
        returns_series = returns_pd.set_index("timestamp")["return"]

        # Prepare benchmark returns if available
        benchmark_returns = None
        if self.benchmark_data is not None and "return" in self.benchmark_data.columns:
            benchmark_pd = self.benchmark_data.to_pandas()
            benchmark_returns = benchmark_pd.set_index("timestamp")["return"]
        elif benchmark_symbol:
            # User can specify a benchmark symbol to use instead
            logging.info(f"Using {benchmark_symbol} as benchmark")

        # Create the tearsheet
        if output_file:
            if output_format.lower() == "html":
                if benchmark_returns is not None:
                    qs.reports.html(
                        returns_series,
                        benchmark_returns,
                        title=title,
                        output=output_file,
                    )
                elif benchmark_symbol:
                    qs.reports.html(
                        returns_series,
                        benchmark_symbol,
                        title=title,
                        output=output_file,
                    )
                else:
                    qs.reports.html(returns_series, title=title, output=output_file)

                logging.info(f"Tearsheet saved to {output_file}")

            elif output_format.lower() == "pdf":
                if benchmark_returns is not None:
                    qs.reports.full(
                        returns_series,
                        benchmark_returns,
                        title=title,
                        output=output_file,
                    )
                elif benchmark_symbol:
                    qs.reports.full(
                        returns_series,
                        benchmark_symbol,
                        title=title,
                        output=output_file,
                    )
                else:
                    qs.reports.full(returns_series, title=title, output=output_file)

                logging.info(f"Tearsheet saved to {output_file}")
            else:
                logging.error(
                    f"Unsupported output format: {output_format}. Use 'html' or 'pdf'."
                )
        else:
            # Display in notebook
            if benchmark_returns is not None:
                qs.reports.html(returns_series, benchmark_returns, title=title)
            elif benchmark_symbol:
                qs.reports.html(returns_series, benchmark_symbol, title=title)
            else:
                qs.reports.html(returns_series, title=title)

    def plot_returns(self, benchmark_symbol: str = None, show: bool = True) -> None:
        """Plot returns using QuantStats.

        Args:
            benchmark_symbol: Ticker symbol for benchmark
            show: Whether to display the plot

        """
        if not HAS_QS:
            logging.error("QuantStats not installed. Cannot plot returns.")
            return

        # Prepare returns series - convert polars DataFrame to pandas Series for QuantStats
        returns_pd = self.returns.to_pandas()
        returns_series = returns_pd.set_index("timestamp")["return"]

        # Plot returns
        qs.plots.returns(returns_series, benchmark=benchmark_symbol, show=show)

    def plot_drawdown(self, show: bool = True) -> None:
        """Plot drawdowns using QuantStats.

        Args:
            show: Whether to display the plot

        """
        if not HAS_QS:
            logging.error("QuantStats not installed. Cannot plot drawdown.")
            return

        # Prepare returns series - convert polars DataFrame to pandas Series for QuantStats
        returns_pd = self.returns.to_pandas()
        returns_series = returns_pd.set_index("timestamp")["return"]

        # Plot drawdown
        qs.plots.drawdown(returns_series, show=show)

    def plot_monthly_returns(self, show: bool = True) -> None:
        """Plot monthly returns heatmap using QuantStats.

        Args:
            show: Whether to display the plot

        """
        if not HAS_QS:
            logging.error("QuantStats not installed. Cannot plot monthly returns.")
            return

        # Prepare returns series - convert polars DataFrame to pandas Series for QuantStats
        returns_pd = self.returns.to_pandas()
        returns_series = returns_pd.set_index("timestamp")["return"]

        # Plot monthly returns
        qs.plots.monthly_returns(returns_series, show=show)

    def export_results(self, format_type: str, file_path: str) -> None:
        """Export backtest results to file.

        Args:
            format_type: Export format ('csv', 'json', 'html', or 'tearsheet')
            file_path: Path to save the exported results

        Raises:
            ValueError: If format_type is not supported

        """
        if format_type.lower() == "csv":
            # Export equity curve and returns
            self.equity_curve.write_csv(f"{file_path}_equity.csv")
            self.returns.write_csv(f"{file_path}_returns.csv")

            # Export statistics as CSV
            stats_df = pl.DataFrame([self.stats])
            stats_df.write_csv(f"{file_path}_stats.csv")

        elif format_type.lower() == "json":
            # Export all results as JSON
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, default=str, indent=2)

        elif format_type.lower() == "html":
            if not HAS_QS:
                raise ValueError("QuantStats is required for HTML export")

            # Create a tearsheet using QuantStats
            self.create_tearsheet(
                title="Backtest Results", output_file=file_path, output_format="html"
            )

        elif format_type.lower() == "tearsheet":
            if not HAS_QS:
                raise ValueError("QuantStats is required for tearsheet export")

            # Create a tearsheet using QuantStats
            self.create_tearsheet(
                title="Backtest Results", output_file=file_path, output_format="html"
            )

        else:
            raise ValueError(
                f"Unsupported export format: {format_type}. Use 'csv', 'json', 'html', or 'tearsheet'."
            )


class Analytics:
    """Analytics for the Vegas backtesting engine.

    This class provides functionality for calculating statistics and
    generating visualizations from backtest results.
    """

    @staticmethod
    def calculate_stats(
        portfolio, broker, start_date, end_date, benchmark_data=None
    ) -> Dict[str, Any]:
        """Calculate performance statistics using QuantStats.

        Args:
            portfolio: Portfolio object
            broker: Broker object
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_data: Optional benchmark data

        Returns:
            Dictionary containing performance statistics

        """
        # Get portfolio metrics
        equity_curve = portfolio.get_equity_curve()
        returns = portfolio.get_returns()

        # Calculate trading metrics
        transactions = broker.transactions
        num_trades = len(transactions)

        if num_trades > 0:
            profitable_trades = sum(1 for t in transactions if t.quantity * t.price > 0)
            win_rate = profitable_trades / num_trades
        else:
            win_rate = 0.0

        # Prepare returns series for QuantStats
        if not returns.is_empty() and HAS_QS:
            # Convert polars DataFrame to pandas Series for QuantStats
            returns_pd = returns.to_pandas()
            returns_series = returns_pd.set_index("timestamp")["return"]

            # Calculate key metrics using QuantStats
            stats = {
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": (end_date - start_date).days,
                "initial_capital": portfolio.initial_cash,
                "final_equity": portfolio.current_equity,
                "total_return": portfolio.current_equity - portfolio.initial_cash,
                "total_return_pct": (
                    (portfolio.current_equity / portfolio.initial_cash) - 1
                )
                * 100,
                "cagr": (
                    qs.stats.cagr(returns_series) * 100
                    if len(returns_series) > 1
                    else 0
                ),
                "sharpe_ratio": (
                    qs.stats.sharpe(returns_series) if len(returns_series) > 1 else 0
                ),
                "sortino_ratio": (
                    qs.stats.sortino(returns_series) if len(returns_series) > 1 else 0
                ),
                "max_drawdown": (
                    qs.stats.max_drawdown(returns_series) * 100
                    if len(returns_series) > 1
                    else 0
                ),
                "volatility_annualized": (
                    qs.stats.volatility(returns_series) * 100
                    if len(returns_series) > 1
                    else 0
                ),
                "num_trades": num_trades,
                "win_rate": win_rate * 100,  # Convert to percentage
            }

            # Add more QuantStats metrics
            if len(returns_series) > 1:
                try:
                    stats.update(
                        {
                            "calmar_ratio": qs.stats.calmar(returns_series),
                            "omega_ratio": qs.stats.omega(returns_series),
                            "skew": qs.stats.skew(returns_series),
                            "kurtosis": qs.stats.kurtosis(returns_series),
                            "var": qs.stats.var(returns_series) * 100,  # Value at Risk
                            "cvar": qs.stats.cvar(returns_series)
                            * 100,  # Conditional VaR
                            "best_day": qs.stats.best(returns_series) * 100,
                            "worst_day": qs.stats.worst(returns_series) * 100,
                            "avg_up_month": qs.stats.avg_win(
                                returns_series, aggregate="M"
                            )
                            * 100,
                            "avg_down_month": qs.stats.avg_loss(
                                returns_series, aggregate="M"
                            )
                            * 100,
                        }
                    )
                except Exception as e:
                    logging.warning(f"Error calculating some QuantStats metrics: {e}")
        else:
            # Fallback to basic metrics if QuantStats not available or returns empty
            stats = {
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": (end_date - start_date).days,
                "initial_capital": portfolio.initial_cash,
                "final_equity": portfolio.current_equity,
                "total_return": portfolio.current_equity - portfolio.initial_cash,
                "total_return_pct": (
                    (portfolio.current_equity / portfolio.initial_cash) - 1
                )
                * 100,
                "num_trades": num_trades,
                "win_rate": win_rate * 100,  # Convert to percentage
            }

        # Add benchmark comparison if available
        if benchmark_data is not None and HAS_QS and not returns.is_empty():
            # Convert polars DataFrame to pandas Series for QuantStats
            benchmark_pd = benchmark_data.to_pandas()
            benchmark_returns = benchmark_pd.set_index("timestamp")["return"]

            try:
                stats.update(
                    {
                        "alpha": qs.stats.alpha(returns_series, benchmark_returns)
                        * 100,
                        "beta": qs.stats.beta(returns_series, benchmark_returns),
                        "correlation": qs.stats.correlation(
                            returns_series, benchmark_returns
                        ),
                        "r_squared": qs.stats.r_squared(
                            returns_series, benchmark_returns
                        ),
                        "information_ratio": qs.stats.information_ratio(
                            returns_series, benchmark_returns
                        ),
                        "treynor_ratio": qs.stats.treynor_ratio(
                            returns_series, benchmark_returns
                        ),
                    }
                )
            except Exception as e:
                logging.warning(f"Error calculating benchmark comparison metrics: {e}")

        return stats

    @staticmethod
    def create_results(portfolio, broker, stats, benchmark_data=None) -> Results:
        """Create backtest results object.

        Args:
            portfolio: Portfolio object
            broker: Broker object
            stats: Dictionary of performance statistics
            benchmark_data: Optional benchmark data

        Returns:
            Results object

        """
        equity_curve = portfolio.get_equity_curve()
        returns = portfolio.get_returns()
        positions_history = portfolio.get_positions_history()

        # Ensure all dataframes are polars DataFrames
        if not isinstance(equity_curve, pl.DataFrame):
            equity_curve = pl.from_pandas(equity_curve)
            
        if not isinstance(returns, pl.DataFrame):
            returns = pl.from_pandas(returns)
            
        if benchmark_data is not None and not isinstance(benchmark_data, pl.DataFrame):
            benchmark_data = pl.from_pandas(benchmark_data)

        return Results(
            equity_curve=equity_curve,
            trades=broker.transactions,
            stats=stats,
            positions_history=positions_history,
            returns=returns,
            benchmark_data=benchmark_data,
        )

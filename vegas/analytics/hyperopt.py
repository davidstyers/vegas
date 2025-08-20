"""Hyperparameter optimization utilities for the Vegas backtesting engine.

This module provides functionality for optimizing strategy parameters using Optuna,
including parameter evaluation, objective functions, and optimization workflows.
"""

import gc
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import optuna
from tabulate import tabulate

from vegas.strategy import Strategy


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Backtest configuration
    initial_capital: float

    # Parameter ranges (must be provided by user)
    param_ranges: Dict[str, Any]
    
    # Evaluation configuration
    objective_metric: str  # Can be "sharpe_ratio", "returns", "max_drawdown", etc.
    n_trials: int
    study_name: str
    direction: str
    n_jobs: int = 1
    invalid_value: float = -1e9  # Value to return for failed evaluations


def evaluate_strategy_params(
    strategy_factory: Callable[..., Strategy],
    strategy_params: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    config: OptimizationConfig,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Evaluate strategy parameters by running a backtest.
    
    Args:
        strategy_factory: Function that creates a strategy instance
        strategy_params: Parameters to pass to the strategy factory
        start_date: Backtest start date
        end_date: Backtest end date
        config: Optimization configuration
        logger: Optional logger instance
        
    Returns:
        Objective metric value from the backtest results
        
    Raises:
        Exception: If backtest fails or returns invalid results
    """
    strategy = None
    engine = None
    results = None
    
    try:
        from vegas.engine import BacktestEngine
        strategy = strategy_factory(**strategy_params)
        engine = BacktestEngine()
        results = engine.run(
            start=start_date,
            end=end_date,
            strategy=strategy,
            initial_capital=config.initial_capital,
        )
        
        stats = results.stats
        
        # Extract the objective metric
        if config.objective_metric == "sharpe_ratio":
            metric_value = float(stats.get("sharpe_ratio", 0.0) or 0.0)
        elif config.objective_metric == "returns":
            metric_value = float(stats.get("total_return", 0.0) or 0.0)
        elif config.objective_metric == "max_drawdown":
            metric_value = -float(stats.get("max_drawdown", 0.0) or 0.0)  # Negative because we minimize drawdown
        elif config.objective_metric == "calmar_ratio":
            metric_value = float(stats.get("calmar_ratio", 0.0) or 0.0)
        elif config.objective_metric == "sortino_ratio":
            metric_value = float(stats.get("sortino_ratio", 0.0) or 0.0)
        else:
            # Default to sharpe ratio if unknown metric
            metric_value = float(stats.get("sharpe_ratio", 0.0) or 0.0)
        
        if logger:
            logger.info(f"{config.objective_metric}: {metric_value}")
            
        if math.isnan(metric_value) or math.isinf(metric_value):
            metric_value = config.invalid_value
            
        return metric_value
        
    except Exception as e:
        if logger:
            logger.error(f"Backtest failed: {e}")
        return config.invalid_value
        
    finally:
        # Clean up resources
        try:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
            
        try:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass
            
        results = None
        engine = None
        strategy = None
        gc.collect()


def create_optuna_objective(
    strategy_factory: Callable[..., Strategy],
    start_date: datetime,
    end_date: datetime,
    config: OptimizationConfig,
    logger: Optional[logging.Logger] = None,
    param_suggestions: Optional[Dict[str, Callable]] = None,
) -> Callable[[optuna.trial.Trial], float]:
    """Create an Optuna objective function for strategy optimization.
    
    Args:
        strategy_factory: Function that creates a strategy instance
        start_date: Backtest start date
        end_date: Backtest end date
        config: Optimization configuration
        logger: Optional logger instance
        param_suggestions: Dictionary mapping parameter names to Optuna suggestion functions
        
    Returns:
        Optuna objective function
    """
    def objective(trial: optuna.trial.Trial) -> float:
        # Use provided parameter suggestions or config parameter ranges
        if param_suggestions:
            params = {}
            for param_name, suggest_func in param_suggestions.items():
                params[param_name] = suggest_func(trial)
        else:
            # Use parameter ranges from config
            params = {}
            for param_name, param_config in config.param_ranges.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"]
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config["low"], 
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config["choices"]
                    )
                elif param_config["type"] == "loguniform":
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"], 
                        log=True
                    )
        
        return evaluate_strategy_params(
            strategy_factory=strategy_factory,
            strategy_params=params,
            start_date=start_date,
            end_date=end_date,
            config=config,
            logger=logger,
        )
    
    return objective


def optimize_strategy(
    strategy_factory: Callable[..., Strategy],
    start_date: datetime,
    end_date: datetime,
    config: OptimizationConfig,
    logger: Optional[logging.Logger] = None,
    param_suggestions: Optional[Dict[str, Callable]] = None,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Optimize strategy parameters using Optuna.
    
    Args:
        strategy_factory: Function that creates a strategy instance
        start_date: Backtest start date
        end_date: Backtest end date
        config: Optimization configuration
        logger: Optional logger instance
        param_suggestions: Dictionary mapping parameter names to Optuna suggestion functions
        
    Returns:
        Tuple of (best_parameters, study)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create objective function
    objective = create_optuna_objective(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        logger=logger,
        param_suggestions=param_suggestions,
    )
    
    # Create and run study
    study = optuna.create_study(
        direction=config.direction,
        study_name=config.study_name,
    )
    
    optimize_kwargs = {
        "n_trials": max(1, int(config.n_trials)),
        "n_jobs": config.n_jobs,
    }
    
    try:
        study.optimize(objective, gc_after_trial=True, **optimize_kwargs)
    except TypeError:
        # Older Optuna versions do not support gc_after_trial
        study.optimize(objective, **optimize_kwargs)
    
    # Get best parameters
    if len(study.best_trials) == 0:
        logger.warning("No successful trials found")
        return {}, study
    
    def trial_key(t: optuna.trial.FrozenTrial) -> float:
        v = t.values or (float("-inf"))
        return float(v[0])
    
    best = max(study.best_trials, key=trial_key)
    best_params = best.params
    
    logger.info(
        "Best parameters: %s | Best value: %.4f",
        best_params,
        float(best.values[0]) if best.values else float("nan"),
    )
    
    return best_params, study


def run_optimized_backtest(
    strategy_factory: Callable[..., Strategy],
    start_date: datetime,
    end_date: datetime,
    config: OptimizationConfig,
    param_suggestions: Optional[Dict[str, Callable]] = None,
) -> Strategy:
    """Run a backtest with optional hyperparameter optimization.
    
    Args:
        strategy_factory: Function that creates a strategy instance
        start_date: Backtest start date
        end_date: Backtest end date
        config: Optimization configuration
        param_suggestions: Dictionary mapping parameter names to Optuna suggestion functions
        
    Returns:
        Backtest results dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Perform optimization
    best_params, study = optimize_strategy(
        strategy_factory=strategy_factory,
        start_date=start_date,
        end_date=end_date,
        config=config,
        logger=logger,
        param_suggestions=param_suggestions,
    )
    
    if not best_params:
        logger.warning("Optimization failed, no valid parameters found")
        return {}
    
    # Run final evaluation with best parameters
    strategy = strategy_factory(**best_params)
    
    return strategy


def print_optimization_summary(
    study: optuna.Study,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print a summary of optimization results.
    
    Args:
        study: Optuna study object
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(study.trials) == 0:
        logger.info("No trials completed")
        return
    
    # Get best trial
    best_trial = study.best_trial
    best_value = best_trial.value if best_trial.value is not None else float("nan")
    best_params = best_trial.params
    
    logger.info("Optimization Summary:")
    logger.info(f"  Total trials: {len(study.trials)}")
    logger.info(f"  Best value: {best_value:.4f}")
    logger.info(f"  Best parameters: {best_params}")
    
    # Print parameter importance if available
    try:
        importance = optuna.importance.get_param_importances(study)
        if importance:
            logger.info("  Parameter importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {param}: {imp:.4f}")
    except Exception as e:
        logger.debug(f"Could not calculate parameter importance: {e}")
    
    # Print optimization history
    logger.info("  Optimization history:")
    for i, trial in enumerate(study.trials[:10]):  # Show first 10 trials
        value = trial.value if trial.value is not None else "N/A"
        logger.info(f"    Trial {i+1}: {value}")
    
    if len(study.trials) > 10:
        logger.info(f"    ... and {len(study.trials) - 10} more trials")

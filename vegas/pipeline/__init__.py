"""Pipeline system for the Vegas backtesting engine.

This module provides a vectorized data processing engine that allows users to define
and compute cross-sectional computations on market data.
"""

# Core pipeline components
from vegas.pipeline.pipeline import Pipeline
from vegas.pipeline.engine import PipelineEngine
from vegas.pipeline.terms import Term, Factor, Filter, Classifier

# Factor classes
from vegas.pipeline.factors.custom import CustomFactor
from vegas.pipeline.factors.basic import (
    Returns,
    SimpleMovingAverage,
    ExponentialWeightedMovingAverage,
    VWAP,
    StandardDeviation,
)
from vegas.pipeline.factors.statistical import ZScore, Rank, Percentile

# Filter classes
from vegas.pipeline.filters.basic import StaticAssets, BinaryCompare, NotNaN
from vegas.pipeline.filters.advanced import All, Any, AtLeastN, NotMissing

# Helper functions
def attach_pipeline(engine, pipeline, name):
    """
    Register a pipeline to be computed before each trading day.
    
    Parameters
    ----------
    engine : BacktestEngine
        The engine to attach the pipeline to
    pipeline : Pipeline
        The pipeline to compute
    name : str
        The name to give the pipeline
        
    Returns
    -------
    Pipeline
        The pipeline that was attached
    """
    return engine.attach_pipeline(pipeline, name)


def pipeline_output(engine, name):
    """
    Get the results of a pipeline for the current day.
    
    Parameters
    ----------
    engine : BacktestEngine
        The engine that contains the pipeline
    name : str
        Name of the pipeline
        
    Returns
    -------
    pd.DataFrame
        The computed pipeline output for the current day
    """
    return engine.pipeline_output(name)

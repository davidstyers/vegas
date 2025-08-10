"""Pipeline class implementation for the Vegas backtesting engine.

This module defines the Pipeline class that represents a collection of computations
to be executed at the start of each trading day.
"""
from typing import Dict, Optional, Any


class Pipeline:
    """
    A container for computations to be executed at the start of each trading day.
    
    A Pipeline represents a collection of computations (Factors, Filters, Classifiers)
    that are executed together to produce a DataFrame of results.
    """
    def __init__(self, columns: Optional[Dict[str, Any]] = None, screen=None, data_portal=None, frequency='1h'):
        """
        Initialize a pipeline with optional columns and screen.
        
        Parameters
        ----------
        columns : dict, optional
            A dictionary mapping column names to terms (Factors, Filters, Classifiers)
        screen : Filter, optional
            A Filter representing assets to include in the computed Pipeline output
        data_portal : DataPortal, optional
            The data portal to use for fetching data
        frequency : str, optional
            The frequency of the data to fetch
        """
        self.columns = columns or {}
        self.screen = screen
        self.data_portal = data_portal
        self.frequency = frequency
        
    def add(self, term, name, overwrite=False):
        """
        Add a term to the pipeline.
        
        Parameters
        ----------
        term : Term
            The term to add to the pipeline.
        name : str
            The name to assign to the term.
        overwrite : bool, optional
            Whether to overwrite an existing term with the same name.
            
        Returns
        -------
        self : Pipeline
            The pipeline with the term added.
            
        Raises
        ------
        KeyError
            If a term with the given name already exists and overwrite is False.
        """
        if name in self.columns and not overwrite:
            raise KeyError(f"Column '{name}' already exists. To overwrite, set overwrite=True.")
        self.columns[name] = term
        return self
        
    def remove(self, name):
        """
        Remove a term from the pipeline.
        
        Parameters
        ----------
        name : str
            The name of the term to remove.
            
        Returns
        -------
        term : Term
            The removed term.
            
        Raises
        ------
        KeyError
            If no term exists with the given name.
        """
        if name not in self.columns:
            raise KeyError(f"No column named '{name}' exists.")
        return self.columns.pop(name)
        
    def set_screen(self, screen, overwrite=False):
        """
        Set a screen on this Pipeline.
        
        Parameters
        ----------
        screen : Filter
            The filter to apply as a screen.
        overwrite : bool, optional
            Whether to overwrite any existing screen.
            
        Returns
        -------
        self : Pipeline
            The pipeline with the screen set.
            
        Raises
        ------
        ValueError
            If a screen already exists and overwrite is False.
        """
        if self.screen is not None and not overwrite:
            raise ValueError("Pipeline already has a screen. To overwrite, set overwrite=True.")
        self.screen = screen
        return self 
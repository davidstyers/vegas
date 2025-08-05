"""Advanced filters for the Vegas pipeline system.

This module defines more complex filters like All, Any, and AtLeastN.
"""
import numpy as np
from vegas.pipeline.terms import Filter, Term


class All(Filter):
    """
    A Filter that requires all inputs to be True.
    
    Parameters
    ----------
    filters : list[Filter]
        The filters to combine with AND logic.
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute the logical AND of all input filters.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Input arrays from the filters
        """
        # Start with all True
        out[:] = True
        
        # Logical AND with each input
        for i in range(len(inputs)):
            out &= inputs[i]


class Any(Filter):
    """
    A Filter that requires at least one input to be True.
    
    Parameters
    ----------
    filters : list[Filter]
        The filters to combine with OR logic.
    """
    
    def __init__(self, *filters):
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute the logical OR of all input filters.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Input arrays from the filters
        """
        # Start with all False
        out[:] = False
        
        # Logical OR with each input
        for i in range(len(inputs)):
            out |= inputs[i]


class AtLeastN(Filter):
    """
    A Filter requiring at least N of the inputs filters to be True.
    
    Parameters
    ----------
    n : int
        Minimum number of filters that must be True.
    filters : list[Filter]
        The filters to check.
    """
    
    def __init__(self, n, *filters):
        self.n = n
        self.filters = filters
        window_length = max([f.window_length for f in filters])
        
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if n > len(filters):
            raise ValueError(f"n ({n}) cannot exceed the number of filters ({len(filters)})")
        
        super().__init__(inputs=list(filters), window_length=window_length)
    
    def compute(self, today, assets, out, *inputs):
        """
        Compute whether at least N of the input filters are True.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        *inputs : tuple of np.array
            Input arrays from the filters
        """
        # Count how many filters are True for each asset
        count = np.zeros_like(out, dtype=int)
        for i in range(len(inputs)):
            count += inputs[i]
        
        # Check if count reaches the threshold
        out[:] = count >= self.n


class NotMissing(Filter):
    """
    A Filter selecting assets with non-missing data.
    
    Parameters
    ----------
    term : Term
        Term to check for missing values.
    """
    
    def __init__(self, term):
        self.term = term
        super().__init__(inputs=[term], window_length=term.window_length)
    
    def compute(self, today, assets, out, data):
        """
        Identify assets with non-missing data.
        
        Parameters
        ----------
        today : pd.Timestamp
            The day for which values are being computed
        assets : np.array
            The assets for which values are requested
        out : np.array[bool]
            Output array of the same shape as assets
        data : np.array
            Data to check for missing values
        """
        # Check if any value in the last row is missing
        if data.ndim > 1:
            out[:] = ~np.isnan(data[-1])
        else:
            out[:] = ~np.isnan(data)


class TopN(Filter):
    """
    A Filter selecting the top N assets by a factor's value on the most recent row.
    
    Parameters
    ----------
    term : Term
        Factor-like term providing numeric values.
    n : int
        Number of assets to select.
    ascending : bool
        If True, select smallest values; if False, select largest values.
    """
    def __init__(self, term: Term, n: int, ascending: bool = False, mask: Filter | None = None):
        if n < 1:
            raise ValueError(f"TopN requires n>=1, got {n}")
        self.term = term
        self.n = int(n)
        self.ascending = bool(ascending)
        # Respect provided mask or term.mask by threading it into the Filter base via self.mask
        super().__init__(inputs=[term], window_length=term.window_length, mask=mask or getattr(term, 'mask', None))
    
    def compute(self, today, assets, out, data):
        """
        Produce a boolean mask for top N selection on the most recent row.
        """
        # Extract the last row
        row = data[-1] if getattr(data, "ndim", 1) > 1 else data

        # Start with all False
        out[:] = False

        # Build candidate mask: finite values only
        finite = np.isfinite(row)

        # Apply term-level mask if present (must match length)
        if self.mask is not None and hasattr(self.mask, 'latest_mask_values'):
            # If engine supports computing mask separately, we'd pass it; since not available here,
            # fall back to finite only. Engine-side will already filter by screen separately.
            pass  # No-op: engine applies screen; intra-filter mask not computed here.

        idx = np.where(finite)[0]
        if idx.size == 0:
            return

        values = row[idx]

        # Determine order for selection
        order = np.argsort(values)  # ascending
        if not self.ascending:
            order = order[::-1]

        take = idx[order[: min(self.n, idx.size)]]
        out[take] = True
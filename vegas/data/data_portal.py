from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Union

import polars as pl
import pytz

try:
    # Optional import to avoid tight coupling; engine passes calendar instance
    from vegas.calendars.base import TradingCalendar  # type: ignore
except Exception:  # pragma: no cover - calendar types are optional here
    TradingCalendar = object  # type: ignore


class DataPortal:
    """
    Centralized, in-memory data cache and access API for backtesting.

    Responsibilities:
    - Bulk-load all required data for a backtest window into memory once.
    - Serve ultra-low latency slices and rolling windows from the in-memory cache.
    - Provide a single source of truth for market data during a run.
    """

    def __init__(self, data_layer):
        self.data_layer = data_layer
        self.timezone = data_layer.timezone
        self._current_dt: Optional[datetime] = None

        # In-memory caches keyed by frequency (e.g., '1h', '1d')
        self._data_by_freq: Dict[str, pl.DataFrame] = {}
        self._loaded_symbols: List[str] = []
        self._loaded_start: Optional[datetime] = None
        self._loaded_end: Optional[datetime] = None
        # Active calendar used to filter timestamps (if any)
        self._calendar: Optional[TradingCalendar] = None

    # -------------------- Lifecycle --------------------
    def load_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[Union[str, List[str]]] = None,
        frequencies: Optional[Iterable[str]] = None,
        market_hours: Optional[tuple] = None,
        calendar: Optional[TradingCalendar] = None,
        data_type: str = "ohlcv",
        limit: Optional[int] = None,
    ) -> None:
        """
        Bulk fetch all required market data into memory for the backtest window.

        Args:
            start_date: Inclusive start timestamp for the backtest.
            end_date: Inclusive end timestamp for the backtest.
            symbols: Optional symbol or list. None means all available symbols.
            frequencies: Iterable of desired bar frequencies to cache (default: {'1h'}).
            market_hours: Deprecated. Ignored when a calendar is provided.
            calendar: Optional TradingCalendar to filter timestamps.
            data_type: Type of data to retrieve ("ohlcv" or "tick")
            limit: Optional limit on number of rows returned (useful for tick data)
        """
        # Normalize arguments
        freq_list = list(frequencies) if frequencies else ["1h"]
        if isinstance(symbols, str):
            sym_list: Optional[List[str]] = [symbols]
        else:
            sym_list = symbols

        # Persist calendar reference and synchronize timezone
        self._calendar = calendar
        if calendar is not None:
            try:
                self.timezone = getattr(calendar, "timezone", self.timezone)
            except Exception:
                pass

        # Pull base/native data once from the DataLayer (do not attempt calendar filtering here)
        # Use the first frequency as the base frequency for data loading
        base_frequency = list(frequencies)[0] if frequencies else "1h"
        
        # If we have an original frequency specification from the engine, use that for transformation
        transform_frequency = getattr(self, '_original_frequency', base_frequency)
        
        # If we have the original frequency spec from the engine, pass it to the data layer
        if hasattr(self, '_original_frequency_spec'):
            self.data_layer._engine_frequency_spec = self._original_frequency_spec
        
        base_df = self.data_layer.get_data_for_backtest(
            start=start_date,
            end=end_date,
            symbols=sym_list,
            market_hours=market_hours,
            data_type=data_type,
            frequency=transform_frequency,
            limit=limit,
        )
        if base_df is None or base_df.is_empty():
            # Clear prior caches to avoid stale state
            self._data_by_freq.clear()
            self._loaded_symbols = []
            self._loaded_start = start_date
            self._loaded_end = end_date
            return

        # Ensure required columns exist and have expected types
        cols = set(base_df.columns)
        required = {"timestamp", "symbol"}
        if not required.issubset(cols):
            raise ValueError(f"Base market data missing required columns: {sorted(required - cols)}")

        # Ensure 'symbol' is Utf8 (string)
        if base_df.get_column("symbol").dtype != pl.Utf8:
            base_df = base_df.with_columns(pl.col("symbol").cast(pl.Utf8))

        # Materialize eager DataFrame (some loaders may return lazy)
        if hasattr(base_df, "collect"):
            try:
                base_df = base_df.collect()
            except Exception:
                # Already eager
                pass

        # Apply calendar-based timestamp filtering if provided
        if calendar is not None:
            try:
                # Filter timestamps vectorially using the calendar
                allowed_ts = calendar.filter_timestamps(base_df.get_column("timestamp"))  # type: ignore[attr-defined]
                if allowed_ts is not None and not allowed_ts.is_empty():
                    base_df = base_df.filter(pl.col("timestamp").is_in(allowed_ts.implode()))
                else:
                    # No timestamps remain under this calendar -> empty cache
                    self._data_by_freq.clear()
                    self._loaded_symbols = []
                    self._loaded_start = start_date
                    self._loaded_end = end_date
                    return
            except Exception:
                # Fail-open: if calendar filtering fails, keep unfiltered data
                pass

        # Store universe and window
        try:
            self._loaded_symbols = (
                base_df.select("symbol").unique().get_column("symbol").to_list()
            )
        except Exception:
            self._loaded_symbols = []
        self._loaded_start = start_date
        self._loaded_end = end_date

        # Always keep the native/hourly dataset available
        self._data_by_freq.clear()
        self._data_by_freq["1h"] = base_df.sort("timestamp")
        
        # If we have an original frequency from the engine, cache under that as well
        original_freq = getattr(self, '_original_frequency', None)
        if original_freq and original_freq != "1h":
            self._data_by_freq[original_freq] = base_df.sort("timestamp")

        # Build additional frequency caches by resampling from the base
        for f in freq_list:
            if f == "1h":
                continue
            self._data_by_freq[f] = self._resample(self._data_by_freq["1h"], f)

    # -------------------- Accessors --------------------
    def set_current_dt(self, dt: datetime) -> None:
        # Normalize to timezone-aware datetime in portal's timezone
        try:
            tz = pytz.timezone(self.timezone)
            if dt.tzinfo is None:
                self._current_dt = tz.localize(dt)
            else:
                self._current_dt = dt.astimezone(tz)
        except Exception:
            self._current_dt = dt

    def get_symbols(self) -> List[str]:
        return list(self._loaded_symbols)

    def get_unified_timestamp_index(self, start: datetime, end: datetime, frequency: str = "1h") -> pl.Series:
        """
        Return unique, sorted timestamps from the in-memory cache for the requested window.
        Falls back to the DataLayer if cache is not populated for that frequency.
        """
        df = self._data_by_freq.get(frequency)
        if df is None or df.is_empty():
            return self.data_layer.get_unified_timestamp_index(start, end)
        # Filter and return unique timestamps
        try:
            tz = self.timezone
            filtered = df.filter(
                (pl.col("timestamp") >= pl.lit(start).cast(pl.Datetime("us", tz)))
                & (pl.col("timestamp") <= pl.lit(end).cast(pl.Datetime("us", tz)))
            )
        except Exception:
            filtered = df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
        if filtered.is_empty():
            return pl.Series("timestamp", [], dtype=pl.Datetime(time_zone=self.timezone))
        return (
            filtered.select(pl.col("timestamp")).unique().sort("timestamp").get_column("timestamp")
        )

    def get_slice(self, timestamp: Optional[datetime] = None, symbols: Optional[Union[str, List[str]]] = None, frequency: str = "1h") -> pl.DataFrame:
        """Alias for get_slice_for_timestamp with frequency selectable."""
        return self.get_slice_for_timestamp(timestamp=timestamp, symbols=symbols, market_hours=None, frequency=frequency)

    def get_slice_for_timestamp(
        self,
        timestamp: Optional[datetime] = None,
        symbols: Optional[Union[str, List[str]]] = None,
        market_hours: Optional[tuple] = None,
        frequency: str = "1h",
    ) -> pl.DataFrame:
        """
        Return all available rows for the given timestamp and optional symbol subset from cache.
        """
        ts = timestamp or self._current_dt
        if ts is None:
            raise ValueError("timestamp is not provided and current_dt is not set")

        df = self._data_by_freq.get(frequency)
        if df is None or df.is_empty():
            return pl.DataFrame()

        # Any market-hours filtering must be applied up-front via calendar in load_data.
        out = df

        # Filter for exact timestamp and optional symbol subset
        out = out.filter(pl.col("timestamp") == ts)
        if isinstance(symbols, str):
            out = out.filter(pl.col("symbol") == symbols)
        elif isinstance(symbols, list) and len(symbols) > 0:
            out = out.filter(pl.col("symbol").is_in(symbols))
        return out

    def get_spot_value(self, asset: str, field: str, dt: datetime, frequency: str = "1h"):
        """Return a scalar field value for a single asset at the given timestamp from cache."""
        df = self.get_slice_for_timestamp(dt, [asset], frequency=frequency)
        if df is None or df.is_empty():
            return None
        try:
            if field in df.columns:
                return df.select(field).to_series().to_list()[-1]
            # Allow 'price' alias to map to 'close' if present
            if field == "price" and "close" in df.columns:
                return df.select("close").to_series().to_list()[-1]
        except Exception:
            return None
        return None

    def history(
        self,
        assets: Optional[Union[str, List[str]]] = None,
        fields: Optional[Union[str, List[str]]] = None,
        bar_count: int = 1,
        frequency: str = "1h",
        end_dt: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Return a rolling window ending at the current_dt from the in-memory cache.
        - assets: None for all symbols; str or list[str] for subset
        - fields: None for all available columns; str or list[str] for subset
        - bar_count: number of trailing bars per symbol
        - frequency: cache frequency to use
        """
        # Resolve end timestamp
        dt = end_dt or self._current_dt
        if dt is None:
            raise ValueError("No end_dt provided and current_dt is not set. Call set_current_dt(dt) or pass end_dt.")

        df = self._data_by_freq.get(frequency)
        if df is None or df.is_empty():
            return pl.DataFrame()

        # Limit to bars up to and including current_dt
        try:
            window = df.filter(pl.col("timestamp") <= pl.lit(dt).cast(pl.Datetime("us", self.timezone)))
        except Exception:
            window = df.filter(pl.col("timestamp") <= dt)
        if isinstance(assets, str):
            window = window.filter(pl.col("symbol") == assets)
        elif isinstance(assets, list) and len(assets) > 0:
            window = window.filter(pl.col("symbol").is_in(assets))
        if window.is_empty():
            return pl.DataFrame()

        # Take the last N rows per symbol
        window = window.sort(["symbol", "timestamp"])
        out = window.group_by("symbol", maintain_order=True).tail(bar_count)

        # Select requested fields plus timestamp/symbol
        if fields is None:
            return out
        req_fields: List[str] = [fields] if isinstance(fields, str) else list(fields)
        # Map 'price' alias to 'close' when present
        req_fields = ["close" if f == "price" else f for f in req_fields]
        keep_cols = [c for c in ["timestamp", "symbol", *req_fields] if c in out.columns]
        return out.select(keep_cols)

    # -------------------- Helpers --------------------
    def _resample(self, df: pl.DataFrame, frequency: str) -> pl.DataFrame:
        """
        Downsample intraday bars to a coarser frequency using OHLCV semantics.
        Only handles time-based frequencies - tick/volume/dollar transformations 
        are handled by the frequency transformation system.
        """
        if df is None or df.is_empty():
            return pl.DataFrame()

        # Check if this is a tick/volume/dollar bar frequency - if so, skip resampling here
        # Handle both "tick:N" format and just "tick" (raw tick data)
        if ":" in frequency:
            bar_type = frequency.split(":")[0].lower()
            if bar_type in ["tick", "volume", "dollar", "tick_imbalance", "volume_imbalance", 
                          "dollar_imbalance", "tick_run", "volume_run", "dollar_run"]:
                # Return the data as-is for non-time frequencies
                # The frequency transformation system will handle these
                return df
        elif frequency.lower() == "tick":
            # Raw tick data - no resampling needed
            return df

        agg_exprs = []
        if "open" in df.columns:
            agg_exprs.append(pl.col("open").first().alias("open"))
        if "high" in df.columns:
            agg_exprs.append(pl.col("high").max().alias("high"))
        if "low" in df.columns:
            agg_exprs.append(pl.col("low").min().alias("low"))
        if "close" in df.columns:
            agg_exprs.append(pl.col("close").last().alias("close"))
        if "volume" in df.columns:
            agg_exprs.append(pl.col("volume").sum().alias("volume"))

        if not agg_exprs:
            # No known OHLCV fields; check if this is a tick-based frequency
            if ":" in frequency:
                bar_type = frequency.split(":")[0].lower()
                if bar_type in ["tick", "volume", "dollar", "tick_imbalance", "volume_imbalance", 
                              "dollar_imbalance", "tick_run", "volume_run", "dollar_run"]:
                    # Return the data as-is for non-time frequencies
                    return df
            elif frequency.lower() == "tick":
                # Raw tick data - no resampling needed
                return df
            
            # For time-based frequencies without OHLCV columns, try time aggregation
            return (
                df.sort("timestamp").group_by_dynamic("timestamp", every=frequency, by="symbol").agg([])
            )

        return (
            df.sort("timestamp").group_by_dynamic("timestamp", every=frequency, by="symbol").agg(agg_exprs)
        )
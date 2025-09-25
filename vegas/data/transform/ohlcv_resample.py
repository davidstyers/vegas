"""OHLCV data resampling and aggregation."""

from typing import List, Optional
import polars as pl
from vegas.data.transform.base import DataTransformer
from vegas.data.transform.frequency_manager import FrequencyManager


class OHLCVResampler(DataTransformer):
    """Resample OHLCV data to different time frequencies.
    
    Supports resampling from higher frequency to lower frequency
    (e.g., 1min -> 5min -> 1h -> 1d).
    """
    
    def __init__(self, target_frequency: str, source_frequency: Optional[str] = None, **kwargs):
        """Initialize OHLCV resampler.
        
        Args:
            target_frequency: Target frequency for resampling (e.g., "1h", "4h", "1d")
            source_frequency: Source frequency (optional, for validation)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.target_frequency = target_frequency
        self.source_frequency = source_frequency
        self.polars_frequency = FrequencyManager.get_polars_time_frequency(target_frequency)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for OHLCV resampling."""
        return ["timestamp", "open", "high", "low", "close", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for OHLCV resampling.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If input is invalid
        """
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        # Check if timestamp column is datetime
        if df["timestamp"].dtype not in [pl.Datetime, pl.Date]:
            raise ValueError("timestamp column must be datetime type")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample OHLCV data to target frequency.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            Resampled DataFrame with OHLCV data at target frequency
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Ensure timestamp is sorted
        df = df.sort("timestamp")
        
        # Handle multiple symbols if present
        if "symbol" in df.columns:
            return self._resample_multi_symbol(df)
        else:
            return self._resample_single_symbol(df)
            
    def _resample_single_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample single symbol OHLCV data.
        
        Args:
            df: DataFrame with single symbol OHLCV data
            
        Returns:
            Resampled DataFrame
        """
        # Use group_by_dynamic for time-based resampling
        resampled = df.group_by_dynamic(
            "timestamp", 
            every=self.polars_frequency,
            closed="left",
            label="left"
        ).agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"), 
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        
        # Filter out periods with no data
        resampled = resampled.filter(pl.col("volume") > 0)
        
        return resampled.sort("timestamp")
        
    def _resample_multi_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample multi-symbol OHLCV data.
        
        Args:
            df: DataFrame with multiple symbols OHLCV data
            
        Returns:
            Resampled DataFrame
        """
        # Group by symbol first, then resample each group
        resampled_groups = []
        
        for symbol, group_df in df.group_by("symbol"):
            symbol_str = symbol[0] if isinstance(symbol, tuple) else symbol
            
            # Resample this symbol's data
            resampled_group = self._resample_single_symbol(
                group_df.drop("symbol")
            ).with_columns(
                pl.lit(symbol_str).alias("symbol")
            )
            
            resampled_groups.append(resampled_group)
            
        if not resampled_groups:
            return pl.DataFrame(schema=df.schema)
            
        # Combine all resampled groups
        result = pl.concat(resampled_groups)
        
        # Reorder columns to match expected format
        column_order = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        existing_cols = [col for col in column_order if col in result.columns]
        
        return result.select(existing_cols).sort(["timestamp", "symbol"])
        
    def can_resample(self, source_freq: str, target_freq: str) -> bool:
        """Check if resampling from source to target frequency is valid.
        
        Args:
            source_freq: Source frequency string
            target_freq: Target frequency string
            
        Returns:
            True if resampling is valid (target >= source)
        """
        try:
            source_seconds = FrequencyManager.time_to_seconds(source_freq)
            target_seconds = FrequencyManager.time_to_seconds(target_freq)
            return target_seconds >= source_seconds
        except ValueError:
            # If we can't parse frequencies, assume it's valid
            return True
            
    def get_supported_frequencies(self) -> List[str]:
        """Get list of supported time frequencies.
        
        Returns:
            List of frequency strings
        """
        return [
            "1min", "2min", "5min", "10min", "15min", "30min",
            "1h", "2h", "3h", "4h", "6h", "8h", "12h", 
            "1d", "2d", "3d", "1w", "2w", "1M"
        ]


class MultiFrequencyResampler(DataTransformer):
    """Resample OHLCV data to multiple frequencies simultaneously."""
    
    def __init__(self, target_frequencies: List[str], **kwargs):
        """Initialize multi-frequency resampler.
        
        Args:
            target_frequencies: List of target frequencies
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        self.target_frequencies = target_frequencies
        self.resamplers = {
            freq: OHLCVResampler(freq) for freq in target_frequencies
        }
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for multi-frequency resampling."""
        return ["timestamp", "open", "high", "low", "close", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input for multi-frequency resampling."""
        return self.resamplers[self.target_frequencies[0]].validate_input(df)
        
    def transform(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Resample to multiple frequencies.
        
        Args:
            df: Input OHLCV DataFrame
            
        Returns:
            Dictionary mapping frequency to resampled DataFrame
        """
        self.validate_input(df)
        
        results = {}
        for freq in self.target_frequencies:
            results[freq] = self.resamplers[freq].transform(df)
            
        return results
        
    def transform_to_single_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample to multiple frequencies and combine into single DataFrame.
        
        Args:
            df: Input OHLCV DataFrame
            
        Returns:
            Combined DataFrame with frequency column
        """
        results = self.transform(df)
        
        combined_dfs = []
        for freq, freq_df in results.items():
            freq_df_with_freq = freq_df.with_columns(
                pl.lit(freq).alias("frequency")
            )
            combined_dfs.append(freq_df_with_freq)
            
        if not combined_dfs:
            return pl.DataFrame()
            
        return pl.concat(combined_dfs).sort(["frequency", "timestamp"])

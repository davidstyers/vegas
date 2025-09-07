"""Tick bar transformations based on Lopez de Prado's methods.

Implements various bar types from "Advances in Financial Machine Learning":
- Tick bars
- Volume bars 
- Dollar bars
- Tick imbalance bars
- Volume imbalance bars
- Dollar imbalance bars
- Tick run bars
- Volume run bars
- Dollar run bars

All implementations use Polars for maximum efficiency.
"""

from typing import List, Optional, Union
import numpy as np
import polars as pl
from vegas.data.transform.base import (
    BarTransformer, 
    ImbalanceBarTransformer, 
    RunBarTransformer
)


class TickBars(BarTransformer):
    """Create tick bars based on number of ticks."""
    
    def __init__(self, bar_size: int = 1000, **kwargs):
        """Initialize tick bars transformer.
        
        Args:
            bar_size: Number of ticks per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for tick bars."""
        return ["timestamp", "price"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for tick bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into tick bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with tick bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Create bar groups based on tick count within each symbol
        df = df.with_columns(
            (pl.arange(0, df.height).over("symbol") // self.bar_size).alias("bar_id")
        )
        
        # Aggregate into bars, preserving symbol column
        bars = df.group_by(["symbol", "bar_id"]).agg([
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume") if "size" in df.columns 
            else pl.lit(self.bar_size).alias("volume"),
        ]).drop("bar_id")
        
        return bars.sort("timestamp")


class VolumeBars(BarTransformer):
    """Create volume bars based on volume thresholds."""
    
    def __init__(self, bar_size: Union[int, float] = 10000, **kwargs):
        """Initialize volume bars transformer.
        
        Args:
            bar_size: Volume threshold per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for volume bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for volume bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into volume bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with volume bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate cumulative volume and bar groups
        df = df.with_columns(
            pl.col("volume").cumsum().alias("cum_volume")
        ).with_columns(
            (pl.col("cum_volume") / self.bar_size).floor().alias("bar_id")
        )
        
        # Aggregate into bars
        bars = df.group_by("bar_id").agg([
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]).drop("bar_id")
        
        return bars.sort("timestamp")


class DollarBars(BarTransformer):
    """Create dollar bars based on dollar volume thresholds."""
    
    def __init__(self, bar_size: Union[int, float] = 100000, **kwargs):
        """Initialize dollar bars transformer.
        
        Args:
            bar_size: Dollar volume threshold per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for dollar bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for dollar bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into dollar bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with dollar bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate dollar volume and cumulative sum
        df = df.with_columns(
            (pl.col("price") * pl.col("volume")).alias("dollar_volume")
        ).with_columns(
            pl.col("dollar_volume").cumsum().alias("cum_dollar_volume")
        ).with_columns(
            (pl.col("cum_dollar_volume") / self.bar_size).floor().alias("bar_id")
        )
        
        # Aggregate into bars
        bars = df.group_by("bar_id").agg([
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]).drop("bar_id")
        
        return bars.sort("timestamp")


class TickImbalanceBars(ImbalanceBarTransformer):
    """Create tick imbalance bars based on tick flow imbalance."""
    
    def __init__(self, bar_size: int = 1000, expected_imbalance: float = 0.0, **kwargs):
        """Initialize tick imbalance bars transformer.
        
        Args:
            bar_size: Expected number of ticks per bar
            expected_imbalance: Expected tick imbalance ratio
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, expected_imbalance, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for tick imbalance bars."""
        return ["timestamp", "price"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for tick imbalance bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into tick imbalance bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with tick imbalance bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule (buy/sell classification)
        df = df.with_columns(
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule")
        )
        
        # Calculate imbalance and create bars
        return self._create_imbalance_bars(df, "tick_rule")
        
    def _create_imbalance_bars(self, df: pl.DataFrame, imbalance_col: str) -> pl.DataFrame:
        """Create imbalance bars based on imbalance column.
        
        Args:
            df: DataFrame with imbalance column
            imbalance_col: Name of column containing imbalance values
            
        Returns:
            DataFrame with imbalance bars
        """
        bars = []
        current_bar_start = 0
        cumulative_theta = 0.0
        
        # Convert to pandas for iterative processing (more efficient for this algorithm)
        df_pd = df.to_pandas()
        
        for i in range(len(df_pd)):
            cumulative_theta += df_pd[imbalance_col].iloc[i]
            
            # Check if bar should close
            expected_imbalance_abs = abs(self.expected_imbalance) * self.bar_size
            if abs(cumulative_theta) >= expected_imbalance_abs or i == len(df_pd) - 1:
                # Create bar from current_bar_start to i
                bar_data = df_pd.iloc[current_bar_start:i+1]
                
                if len(bar_data) > 0:
                    bar = {
                        "timestamp": bar_data["timestamp"].iloc[0],
                        "open": bar_data["price"].iloc[0],
                        "high": bar_data["price"].max(),
                        "low": bar_data["price"].min(),
                        "close": bar_data["price"].iloc[-1],
                        "volume": len(bar_data) if "volume" not in bar_data.columns 
                                 else bar_data["volume"].sum(),
                    }
                    bars.append(bar)
                
                # Reset for next bar
                current_bar_start = i + 1
                cumulative_theta = 0.0
                
        # Convert back to Polars DataFrame
        if bars:
            return pl.DataFrame(bars).sort("timestamp")
        else:
            return pl.DataFrame(schema=self.get_output_schema())


class VolumeImbalanceBars(ImbalanceBarTransformer):
    """Create volume imbalance bars based on volume flow imbalance."""
    
    def __init__(self, bar_size: Union[int, float] = 10000, expected_imbalance: float = 0.0, **kwargs):
        """Initialize volume imbalance bars transformer.
        
        Args:
            bar_size: Expected volume per bar
            expected_imbalance: Expected volume imbalance ratio
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, expected_imbalance, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for volume imbalance bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for volume imbalance bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into volume imbalance bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with volume imbalance bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule and volume imbalance
        df = df.with_columns([
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule"),
        ]).with_columns(
            (pl.col("tick_rule") * pl.col("volume")).alias("volume_imbalance")
        )
        
        # Create imbalance bars
        return self._create_imbalance_bars(df, "volume_imbalance")


class DollarImbalanceBars(ImbalanceBarTransformer):
    """Create dollar imbalance bars based on dollar volume flow imbalance."""
    
    def __init__(self, bar_size: Union[int, float] = 100000, expected_imbalance: float = 0.0, **kwargs):
        """Initialize dollar imbalance bars transformer.
        
        Args:
            bar_size: Expected dollar volume per bar
            expected_imbalance: Expected dollar imbalance ratio
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, expected_imbalance, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for dollar imbalance bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for dollar imbalance bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into dollar imbalance bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with dollar imbalance bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule and dollar imbalance
        df = df.with_columns([
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule"),
            (pl.col("price") * pl.col("volume")).alias("dollar_volume")
        ]).with_columns(
            (pl.col("tick_rule") * pl.col("dollar_volume")).alias("dollar_imbalance")
        )
        
        # Create imbalance bars
        return self._create_imbalance_bars(df, "dollar_imbalance")


class TickRunBars(RunBarTransformer):
    """Create tick run bars based on sequences of buy/sell ticks."""
    
    def __init__(self, bar_size: int = 1000, **kwargs):
        """Initialize tick run bars transformer.
        
        Args:
            bar_size: Expected number of ticks per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for tick run bars."""
        return ["timestamp", "price"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for tick run bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into tick run bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with tick run bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule and runs
        df = df.with_columns(
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule")
        ).with_columns(
            self._calculate_runs(pl.col("tick_rule")).alias("run_group")
        )
        
        # Create run bars
        return self._create_run_bars(df, "tick_rule")
        
    def _create_run_bars(self, df: pl.DataFrame, value_col: str) -> pl.DataFrame:
        """Create run bars based on run groups.
        
        Args:
            df: DataFrame with run groups
            value_col: Column to use for run values
            
        Returns:
            DataFrame with run bars
        """
        bars = []
        
        # Convert to pandas for iterative processing
        df_pd = df.to_pandas()
        
        # Group by run_group and process each run
        for run_id, run_data in df_pd.groupby("run_group"):
            run_length = len(run_data)
            
            # Create bars from this run if it meets the threshold
            if run_length >= self.bar_size:
                bar = {
                    "timestamp": run_data["timestamp"].iloc[0],
                    "open": run_data["price"].iloc[0],
                    "high": run_data["price"].max(),
                    "low": run_data["price"].min(),
                    "close": run_data["price"].iloc[-1],
                    "volume": len(run_data) if "volume" not in run_data.columns 
                             else run_data["volume"].sum(),
                }
                bars.append(bar)
                
        # Convert back to Polars DataFrame
        if bars:
            return pl.DataFrame(bars).sort("timestamp")
        else:
            return pl.DataFrame(schema=self.get_output_schema())


class VolumeRunBars(RunBarTransformer):
    """Create volume run bars based on sequences of buy/sell volume."""
    
    def __init__(self, bar_size: Union[int, float] = 10000, **kwargs):
        """Initialize volume run bars transformer.
        
        Args:
            bar_size: Expected volume per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for volume run bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for volume run bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into volume run bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with volume run bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule, volume imbalance, and runs
        df = df.with_columns([
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule"),
        ]).with_columns([
            (pl.col("tick_rule") * pl.col("volume")).alias("volume_imbalance"),
            self._calculate_runs(pl.col("tick_rule")).alias("run_group")
        ])
        
        # Create run bars using volume imbalance
        return self._create_volume_run_bars(df)
        
    def _create_volume_run_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create volume run bars based on volume runs.
        
        Args:
            df: DataFrame with volume runs
            
        Returns:
            DataFrame with volume run bars
        """
        bars = []
        
        # Convert to pandas for iterative processing
        df_pd = df.to_pandas()
        
        # Group by run_group and process each run
        for run_id, run_data in df_pd.groupby("run_group"):
            run_volume = run_data["volume"].sum()
            
            # Create bars from this run if it meets the volume threshold
            if run_volume >= self.bar_size:
                bar = {
                    "timestamp": run_data["timestamp"].iloc[0],
                    "open": run_data["price"].iloc[0],
                    "high": run_data["price"].max(),
                    "low": run_data["price"].min(),
                    "close": run_data["price"].iloc[-1],
                    "volume": run_data["volume"].sum(),
                }
                bars.append(bar)
                
        # Convert back to Polars DataFrame
        if bars:
            return pl.DataFrame(bars).sort("timestamp")
        else:
            return pl.DataFrame(schema=self.get_output_schema())


class DollarRunBars(RunBarTransformer):
    """Create dollar run bars based on sequences of buy/sell dollar volume."""
    
    def __init__(self, bar_size: Union[int, float] = 100000, **kwargs):
        """Initialize dollar run bars transformer.
        
        Args:
            bar_size: Expected dollar volume per bar
            **kwargs: Additional configuration parameters
        """
        super().__init__(bar_size, **kwargs)
        
    def get_required_columns(self) -> List[str]:
        """Get required columns for dollar run bars."""
        return ["timestamp", "price", "volume"]
        
    def validate_input(self, df: pl.DataFrame) -> bool:
        """Validate input DataFrame for dollar run bars."""
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if df.is_empty():
            raise ValueError("Input DataFrame is empty")
            
        return True
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform tick data into dollar run bars.
        
        Args:
            df: Input tick DataFrame
            
        Returns:
            DataFrame with dollar run bars
        """
        self.validate_input(df)
        
        if df.is_empty():
            return df
            
        # Calculate tick rule, dollar volume, and runs
        df = df.with_columns([
            self._calculate_tick_rule(pl.col("price")).alias("tick_rule"),
            (pl.col("price") * pl.col("volume")).alias("dollar_volume")
        ]).with_columns([
            (pl.col("tick_rule") * pl.col("dollar_volume")).alias("dollar_imbalance"),
            self._calculate_runs(pl.col("tick_rule")).alias("run_group")
        ])
        
        # Create run bars using dollar volume
        return self._create_dollar_run_bars(df)
        
    def _create_dollar_run_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create dollar run bars based on dollar volume runs.
        
        Args:
            df: DataFrame with dollar volume runs
            
        Returns:
            DataFrame with dollar run bars
        """
        bars = []
        
        # Convert to pandas for iterative processing
        df_pd = df.to_pandas()
        
        # Group by run_group and process each run
        for run_id, run_data in df_pd.groupby("run_group"):
            run_dollar_volume = run_data["dollar_volume"].sum()
            
            # Create bars from this run if it meets the dollar volume threshold
            if run_dollar_volume >= self.bar_size:
                bar = {
                    "timestamp": run_data["timestamp"].iloc[0],
                    "open": run_data["price"].iloc[0],
                    "high": run_data["price"].max(),
                    "low": run_data["price"].min(),
                    "close": run_data["price"].iloc[-1],
                    "volume": run_data["volume"].sum(),
                }
                bars.append(bar)
                
        # Convert back to Polars DataFrame
        if bars:
            return pl.DataFrame(bars).sort("timestamp")
        else:
            return pl.DataFrame(schema=self.get_output_schema())

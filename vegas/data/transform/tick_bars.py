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
- Side-based bars (using trade aggressor information)

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
        return ["timestamp", "price", "size"]
        
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
        
        # Prepare aggregation columns
        agg_columns = [
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
        ]
        
        # Add side-based aggregations if side column exists
        if "side" in df.columns:
            # Ensure side column has proper data type for string operations
            df = df.with_columns(
                pl.col("side").cast(pl.Utf8, strict=False)
            )
            agg_columns.extend(self._get_side_aggregations())
        
        # Aggregate into bars, preserving symbol column
        bars = df.group_by(["symbol", "bar_id"]).agg(agg_columns).drop("bar_id")
        
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
        return ["timestamp", "price", "size"]
        
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
            
        # Calculate cumulative volume and bar groups per symbol
        df = df.with_columns(
            pl.col("size").cum_sum().over("symbol").alias("cum_volume")
        ).with_columns(
            (pl.col("cum_volume") / self.bar_size).floor().alias("bar_id")
        )
        
        # Prepare aggregation columns
        agg_columns = [
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
        ]
        
        # Add side-based aggregations if side column exists
        if "side" in df.columns:
            # Ensure side column has proper data type for string operations
            df = df.with_columns(
                pl.col("side").cast(pl.Utf8, strict=False)
            )
            agg_columns.extend(self._get_side_aggregations())
        
        # Aggregate into bars, preserving symbol column
        bars = df.group_by(["symbol", "bar_id"]).agg(agg_columns).drop("bar_id")
        
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
        return ["timestamp", "price", "size"]
        
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
            
        # Calculate dollar volume and cumulative sum per symbol
        df = df.with_columns(
            (pl.col("price") * pl.col("size")).alias("dollar_volume")
        ).with_columns(
            pl.col("dollar_volume").cum_sum().over("symbol").alias("cum_dollar_volume")
        ).with_columns(
            (pl.col("cum_dollar_volume") / self.bar_size).floor().alias("bar_id")
        )
        
        # Prepare aggregation columns
        agg_columns = [
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
        ]
        
        # Add side-based aggregations if side column exists
        if "side" in df.columns:
            # Ensure side column has proper data type for string operations
            df = df.with_columns(
                pl.col("side").cast(pl.Utf8, strict=False)
            )
            agg_columns.extend(self._get_side_aggregations())
        
        # Aggregate into bars, preserving symbol column
        bars = df.group_by(["symbol", "bar_id"]).agg(agg_columns).drop("bar_id")
        
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
            
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based imbalance (buy vs sell aggressors)
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_imbalance"),
            ])
            imbalance_col = "side_imbalance"
        else:
            # Calculate tick rule (buy/sell classification) per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns(
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .when(pl.col("price_diff").is_null()).then(0)  # First tick
                  .otherwise(0)  # No change
                  .alias("tick_rule")
            ).drop("price_diff")
            imbalance_col = "tick_rule"
        
        # Calculate imbalance and create bars
        return self._create_imbalance_bars(df, imbalance_col)


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
        return ["timestamp", "price", "size"]
        
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
            
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based volume imbalance
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_rule"),
            ]).with_columns(
                (pl.col("side_rule") * pl.col("size")).alias("volume_imbalance")
            )
        else:
            # Calculate tick rule and volume imbalance per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns([
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .when(pl.col("price_diff").is_null()).then(0)  # First tick
                  .otherwise(0)  # No change
                  .alias("tick_rule"),
            ]).with_columns(
                (pl.col("tick_rule") * pl.col("size")).alias("volume_imbalance")
            ).drop("price_diff")
        
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
        return ["timestamp", "price", "size"]
        
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
            
        # Calculate dollar volume first
        df = df.with_columns(
            (pl.col("price") * pl.col("size")).alias("dollar_volume")
        )
        
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based dollar imbalance
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_rule"),
            ]).with_columns(
                (pl.col("side_rule") * pl.col("dollar_volume")).alias("dollar_imbalance")
            )
        else:
            # Calculate tick rule and dollar imbalance per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns([
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .when(pl.col("price_diff").is_null()).then(0)  # First tick
                  .otherwise(0)  # No change
                  .alias("tick_rule"),
            ]).with_columns(
                (pl.col("tick_rule") * pl.col("dollar_volume")).alias("dollar_imbalance")
            ).drop("price_diff")
        
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
            
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based rule
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_rule"),
            ]).with_columns(
                (pl.col("side_rule") != pl.col("side_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            )
            rule_col = "side_rule"
        else:
            # Calculate tick rule and runs per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns([
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .otherwise(0)
                  .fill_null(strategy="forward")
                  .fill_null(0)
                  .alias("tick_rule"),
            ]).with_columns(
                (pl.col("tick_rule") != pl.col("tick_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            ).drop("price_diff")
            rule_col = "tick_rule"
        
        # Create run bars
        return self._create_run_bars(df, rule_col)
        
    def _create_run_bars(self, df: pl.DataFrame, value_col: str) -> pl.DataFrame:
        """Create run bars based on run groups.
        
        Args:
            df: DataFrame with run groups
            value_col: Column to use for run values
            
        Returns:
            DataFrame with run bars
        """
        if df.is_empty():
            return pl.DataFrame(schema=self.get_output_schema())
        
        # Filter runs that meet the threshold and aggregate using pure Polars
        bars = (df
                .group_by(["symbol", "run_group"])
                .agg([
                    pl.len().alias("run_length"),
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.when(pl.col("size").is_not_null().any())
                      .then(pl.col("size").sum())
                      .otherwise(pl.len())
                      .alias("volume")
                ])
                .filter(pl.col("run_length") >= self.bar_size)
                .drop("run_length")
                .sort("timestamp"))
        
        return bars


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
        return ["timestamp", "price", "size"]
        
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
            
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based rule
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_rule"),
            ]).with_columns([
                (pl.col("side_rule") * pl.col("size")).alias("volume_imbalance"),
                (pl.col("side_rule") != pl.col("side_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            ])
        else:
            # Calculate tick rule, volume imbalance, and runs per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns([
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .otherwise(0)
                  .fill_null(strategy="forward")
                  .fill_null(0)
                  .alias("tick_rule"),
            ]).with_columns([
                (pl.col("tick_rule") * pl.col("size")).alias("volume_imbalance"),
                (pl.col("tick_rule") != pl.col("tick_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            ]).drop("price_diff")
        
        # Create run bars using volume imbalance
        return self._create_volume_run_bars(df)
        
    def _create_volume_run_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create volume run bars based on volume runs.
        
        Args:
            df: DataFrame with volume runs
            
        Returns:
            DataFrame with volume run bars
        """
        if df.is_empty():
            return pl.DataFrame(schema=self.get_output_schema())
        
        # Filter runs that meet the volume threshold and aggregate using pure Polars
        bars = (df
                .group_by(["symbol", "run_group"])
                .agg([
                    pl.col("size").sum().alias("run_volume"),
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("size").sum().alias("volume")
                ])
                .filter(pl.col("run_volume") >= self.bar_size)
                .drop("run_volume")
                .sort("timestamp"))
        
        return bars


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
        return ["timestamp", "price", "size"]
        
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
            
        # Calculate dollar volume first
        df = df.with_columns(
            (pl.col("price") * pl.col("size")).alias("dollar_volume")
        )
        
        # Use side field if available, otherwise fall back to tick rule
        if "side" in df.columns:
            # Use side-based rule
            df = df.with_columns([
                pl.when(pl.col("side") == "Bid").then(1)
                  .when(pl.col("side") == "Ask").then(-1)
                  .otherwise(0)  # Unknown side
                  .alias("side_rule"),
            ]).with_columns([
                (pl.col("side_rule") * pl.col("dollar_volume")).alias("dollar_imbalance"),
                (pl.col("side_rule") != pl.col("side_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            ])
        else:
            # Calculate tick rule, dollar volume, and runs per symbol
            df = df.with_columns(
                pl.col("price").diff().over("symbol").alias("price_diff")
            ).with_columns([
                pl.when(pl.col("price_diff") > 0).then(1)
                  .when(pl.col("price_diff") < 0).then(-1)
                  .otherwise(0)
                  .fill_null(strategy="forward")
                  .fill_null(0)
                  .alias("tick_rule"),
            ]).with_columns([
                (pl.col("tick_rule") * pl.col("dollar_volume")).alias("dollar_imbalance"),
                (pl.col("tick_rule") != pl.col("tick_rule").shift(1)).cast(pl.Int32).cum_sum().over("symbol").alias("run_group")
            ]).drop("price_diff")
        
        # Create run bars using dollar volume
        return self._create_dollar_run_bars(df)
        
    def _create_dollar_run_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create dollar run bars based on dollar volume runs.
        
        Args:
            df: DataFrame with dollar volume runs
            
        Returns:
            DataFrame with dollar run bars
        """
        if df.is_empty():
            return pl.DataFrame(schema=self.get_output_schema())
        
        # Filter runs that meet the dollar volume threshold and aggregate using pure Polars
        bars = (df
                .group_by(["symbol", "run_group"])
                .agg([
                    pl.col("dollar_volume").sum().alias("run_dollar_volume"),
                    pl.col("timestamp").first().alias("timestamp"),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("size").sum().alias("volume")
                ])
                .filter(pl.col("run_dollar_volume") >= self.bar_size)
                .drop("run_dollar_volume")
                .sort("timestamp"))
        
        return bars

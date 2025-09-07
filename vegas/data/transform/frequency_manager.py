"""Frequency management for data transformations."""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from vegas.data.transform.base import FrequencyType, DataType


@dataclass
class FrequencySpec:
    """Specification for data frequency and transformation."""
    frequency_type: FrequencyType
    value: Union[int, float, str]
    data_type: DataType
    
    def __str__(self) -> str:
        """String representation of frequency spec."""
        return f"{self.frequency_type.value}:{self.value}"


class FrequencyManager:
    """Manager for parsing and handling data frequencies."""
    
    # Time frequency patterns
    TIME_PATTERNS = {
        r"(\d+)s": ("second", 1),
        r"(\d+)min": ("minute", 60), 
        r"(\d+)m": ("minute", 60),
        r"(\d+)h": ("hour", 3600),
        r"(\d+)H": ("hour", 3600),
        r"(\d+)d": ("day", 86400),
        r"(\d+)D": ("day", 86400),
        r"(\d+)w": ("week", 604800),
        r"(\d+)W": ("week", 604800),
    }
    
    # Default bar sizes for different frequency types
    DEFAULT_BAR_SIZES = {
        FrequencyType.TICK: 1000,
        FrequencyType.VOLUME: 10000,
        FrequencyType.DOLLAR: 100000,
        FrequencyType.TICK_IMBALANCE: 1000,
        FrequencyType.VOLUME_IMBALANCE: 10000,
        FrequencyType.DOLLAR_IMBALANCE: 100000,
        FrequencyType.TICK_RUN: 1000,
        FrequencyType.VOLUME_RUN: 10000,
        FrequencyType.DOLLAR_RUN: 100000,
    }
    
    @classmethod
    def parse_frequency(cls, frequency: str, data_type: DataType = DataType.OHLCV) -> FrequencySpec:
        """Parse frequency string into FrequencySpec.
        
        Args:
            frequency: Frequency string (e.g., "1h", "tick:1000", "volume:5000")
            data_type: Type of underlying data
            
        Returns:
            FrequencySpec object
            
        Examples:
            >>> FrequencyManager.parse_frequency("1h")
            FrequencySpec(frequency_type=TIME, value="1h", data_type=OHLCV)
            
            >>> FrequencyManager.parse_frequency("tick:1000")
            FrequencySpec(frequency_type=TICK, value=1000, data_type=TICK)
            
            >>> FrequencyManager.parse_frequency("volume:5000") 
            FrequencySpec(frequency_type=VOLUME, value=5000, data_type=TICK)
        """
        frequency = frequency.strip().lower()
        
        # Check for explicit bar type specification (e.g., "tick:1000")
        if ":" in frequency:
            bar_type, value_str = frequency.split(":", 1)
            
            # Map bar type string to FrequencyType
            bar_type_map = {
                "tick": FrequencyType.TICK,
                "volume": FrequencyType.VOLUME, 
                "dollar": FrequencyType.DOLLAR,
                "tick_imbalance": FrequencyType.TICK_IMBALANCE,
                "volume_imbalance": FrequencyType.VOLUME_IMBALANCE,
                "dollar_imbalance": FrequencyType.DOLLAR_IMBALANCE,
                "tick_run": FrequencyType.TICK_RUN,
                "volume_run": FrequencyType.VOLUME_RUN,
                "dollar_run": FrequencyType.DOLLAR_RUN,
            }
            
            if bar_type not in bar_type_map:
                raise ValueError(f"Unknown bar type: {bar_type}")
                
            frequency_type = bar_type_map[bar_type]
            
            # Parse value
            try:
                if frequency_type in [FrequencyType.DOLLAR, FrequencyType.DOLLAR_IMBALANCE, FrequencyType.DOLLAR_RUN]:
                    value = float(value_str)
                else:
                    value = int(value_str)
            except ValueError:
                raise ValueError(f"Invalid value for {bar_type}: {value_str}")
                
            # For bar types, data type should be TICK
            effective_data_type = DataType.TICK if frequency_type != FrequencyType.TIME else data_type
            
            return FrequencySpec(
                frequency_type=frequency_type,
                value=value,
                data_type=effective_data_type
            )
        
        # Check for time-based frequency patterns
        for pattern, (unit, seconds) in cls.TIME_PATTERNS.items():
            match = re.match(pattern, frequency)
            if match:
                return FrequencySpec(
                    frequency_type=FrequencyType.TIME,
                    value=frequency,
                    data_type=data_type
                )
        
        # Default case - assume it's a time frequency
        return FrequencySpec(
            frequency_type=FrequencyType.TIME,
            value=frequency,
            data_type=data_type
        )
    
    @classmethod
    def get_polars_time_frequency(cls, frequency: str) -> str:
        """Convert frequency string to Polars-compatible time frequency.
        
        Args:
            frequency: Input frequency string
            
        Returns:
            Polars-compatible frequency string
        """
        # Polars uses slightly different notation
        polars_map = {
            "min": "m",
            "hour": "h", 
            "day": "d",
            "week": "w"
        }
        
        for pattern, (unit, seconds) in cls.TIME_PATTERNS.items():
            match = re.match(pattern, frequency.lower())
            if match:
                value = match.group(1)
                polars_unit = polars_map.get(unit, unit[0])
                return f"{value}{polars_unit}"
                
        return frequency
    
    @classmethod
    def time_to_seconds(cls, frequency: str) -> int:
        """Convert time frequency to seconds.
        
        Args:
            frequency: Time frequency string (e.g., "1h", "30min")
            
        Returns:
            Number of seconds
        """
        for pattern, (unit, seconds_per_unit) in cls.TIME_PATTERNS.items():
            match = re.match(pattern, frequency.lower())
            if match:
                value = int(match.group(1))
                return value * seconds_per_unit
                
        raise ValueError(f"Cannot parse time frequency: {frequency}")
    
    @classmethod
    def get_transformer_class(cls, frequency_spec: FrequencySpec):
        """Get appropriate transformer class for frequency specification.
        
        Args:
            frequency_spec: Frequency specification
            
        Returns:
            Transformer class
        """
        from vegas.data.transform.ohlcv_resample import OHLCVResampler
        from vegas.data.transform.tick_bars import (
            TickBars, VolumeBars, DollarBars,
            TickImbalanceBars, VolumeImbalanceBars, DollarImbalanceBars,
            TickRunBars, VolumeRunBars, DollarRunBars
        )
        
        transformer_map = {
            FrequencyType.TIME: OHLCVResampler,
            FrequencyType.TICK: TickBars,
            FrequencyType.VOLUME: VolumeBars,
            FrequencyType.DOLLAR: DollarBars,
            FrequencyType.TICK_IMBALANCE: TickImbalanceBars,
            FrequencyType.VOLUME_IMBALANCE: VolumeImbalanceBars, 
            FrequencyType.DOLLAR_IMBALANCE: DollarImbalanceBars,
            FrequencyType.TICK_RUN: TickRunBars,
            FrequencyType.VOLUME_RUN: VolumeRunBars,
            FrequencyType.DOLLAR_RUN: DollarRunBars,
        }
        
        return transformer_map.get(frequency_spec.frequency_type)
    
    @classmethod
    def validate_frequency_for_data_type(cls, frequency_spec: FrequencySpec) -> bool:
        """Validate that frequency is compatible with data type.
        
        Args:
            frequency_spec: Frequency specification to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Time-based frequencies work with OHLCV data
        if frequency_spec.frequency_type == FrequencyType.TIME:
            return frequency_spec.data_type == DataType.OHLCV
            
        # Bar-based frequencies require tick data
        return frequency_spec.data_type in [DataType.TICK, DataType.TBBO]
    
    @classmethod
    def get_available_frequencies(cls, data_type: DataType) -> Dict[str, str]:
        """Get available frequencies for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Dictionary mapping frequency names to descriptions
        """
        if data_type == DataType.OHLCV:
            return {
                "1min": "1-minute bars",
                "5min": "5-minute bars", 
                "15min": "15-minute bars",
                "1h": "1-hour bars",
                "4h": "4-hour bars",
                "1d": "Daily bars",
                "1w": "Weekly bars",
            }
        elif data_type in [DataType.TICK, DataType.TBBO]:
            return {
                "tick:1000": "1000-tick bars",
                "volume:10000": "10K volume bars",
                "dollar:100000": "100K dollar bars", 
                "tick_imbalance:1000": "1000-tick imbalance bars",
                "volume_imbalance:10000": "10K volume imbalance bars",
                "dollar_imbalance:100000": "100K dollar imbalance bars",
                "tick_run:1000": "1000-tick run bars",
                "volume_run:10000": "10K volume run bars", 
                "dollar_run:100000": "100K dollar run bars",
            }
        else:
            return {}

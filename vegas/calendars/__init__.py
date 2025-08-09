from __future__ import annotations

from typing import Dict

from .base import TradingCalendar, TwentyFourSevenCalendar
try:
    from .nyse import NYSECalendar  # optional, create if not present
except Exception:  # pragma: no cover - optional calendar may be missing in some installs
    NYSECalendar = None  # type: ignore

_CALENDAR_REGISTRY: Dict[str, TradingCalendar] = {}
if NYSECalendar is not None:
    _CALENDAR_REGISTRY.update({
        "NYSE": NYSECalendar(),
        "XNYS": NYSECalendar(),
    })
_CALENDAR_REGISTRY.update({
    "24/7": TwentyFourSevenCalendar(),
    "24/7_CRYPTO": TwentyFourSevenCalendar(name="24/7_CRYPTO"),
})


def get_calendar(name: str) -> TradingCalendar:
    """Return a calendar instance by common name.

    Supported names include "NYSE" and "XNYS".
    """
    key = name.upper()
    if key not in _CALENDAR_REGISTRY:
        supported = ", ".join(sorted(_CALENDAR_REGISTRY))
        raise KeyError(f"Unknown calendar '{name}'. Supported: {supported}")
    return _CALENDAR_REGISTRY[key]


__all__ = [
    "TradingCalendar",
    "NYSECalendar",
    "get_calendar",
]


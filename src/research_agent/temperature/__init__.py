"""Deterministic temperature analysis from ERA5 (no LLM). Errors are the rainfall package's."""

from research_agent.temperature.client import TemperatureArchive, TemperatureClient
from research_agent.temperature.models import (
    AnnualExtreme,
    DayContext,
    ExtremeBaseline,
    MeanContext,
    MonthAnomaly,
    PeriodMean,
    TempBaseline,
    TemperatureExtremes,
    TemperatureMonthly,
    TemperatureRecord,
    TemperatureYearly,
)

__all__ = [
    "AnnualExtreme",
    "DayContext",
    "ExtremeBaseline",
    "MeanContext",
    "MonthAnomaly",
    "PeriodMean",
    "TempBaseline",
    "TemperatureArchive",
    "TemperatureClient",
    "TemperatureExtremes",
    "TemperatureMonthly",
    "TemperatureRecord",
    "TemperatureYearly",
]

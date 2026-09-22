"""Task clocks independent of order pricing and fill assumptions."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class ScheduleConfig:
    cold_start_seconds: float = 10.0
    observation_seconds: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    task_step_seconds: float = 1.0
    task_range_seconds: float = 60.0

    def __post_init__(self):
        for name in ("cold_start_seconds", "task_step_seconds", "task_range_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.cold_start_seconds < self.task_range_seconds:
            raise ValueError("cold_start_seconds must lie in [0, task_range_seconds)")
        if self.task_step_seconds <= 0 or self.task_range_seconds <= 0:
            raise ValueError("task_step_seconds and task_range_seconds must be positive")
        if (not self.observation_seconds
            or any(isinstance(value, bool) or not math.isfinite(value) or value < 0
                   for value in self.observation_seconds)
            or len(set(self.observation_seconds)) != len(self.observation_seconds)):
            raise ValueError("observation_seconds must be unique finite nonnegative durations")
        for value in (self.cold_start_seconds, self.task_step_seconds,
                      self.task_range_seconds, *self.observation_seconds):
            if not math.isclose(value * 1e9, round(value * 1e9), abs_tol=1e-4, rel_tol=0):
                raise ValueError("Time settings must be representable in whole nanoseconds")


def build_task_schedule(open_time: pd.Timestamp, config: ScheduleConfig) -> pd.DataFrame:
    """One task per fixed clock tick after cold start, expanded by observation case.

    The task clock remains anchored to the configured opening. Observations before
    cold start are not discarded. This function creates nominal clocks only; data
    availability, fill rules and any signal-end boundary are separate checks.
    """
    open_time = pd.Timestamp(open_time)
    if pd.isna(open_time) or open_time.tz is None:
        raise ValueError("open_time must be timezone-aware and nonmissing")
    step_ns = round(config.task_step_seconds * 1e9)
    end_ns = round(config.task_range_seconds * 1e9)
    cold_ns = round(config.cold_start_seconds * 1e9)
    first_ns = ((cold_ns + step_ns - 1) // step_ns) * step_ns
    rows = []
    for offset_ns in range(first_ns, end_ns, step_ns):
        task_time = open_time + pd.Timedelta(offset_ns, unit="ns")
        for observation in config.observation_seconds:
            decision = task_time + pd.Timedelta(round(observation * 1e9), unit="ns")
            rows.append({
                "open_time": open_time,
                "task_time": task_time,
                "task_elapsed_seconds": offset_ns / 1e9,
                "cold_start_seconds": config.cold_start_seconds,
                "observation_seconds": observation,
                "decision_time": decision,
                "decision_elapsed_seconds": (decision - open_time).total_seconds(),
                "decision_within_first_minute": decision < open_time + pd.Timedelta(end_ns, unit="ns"),
            })
    columns = ["open_time", "task_time", "task_elapsed_seconds", "cold_start_seconds",
               "observation_seconds", "decision_time", "decision_elapsed_seconds",
               "decision_within_first_minute"]
    return pd.DataFrame(rows, columns=columns)

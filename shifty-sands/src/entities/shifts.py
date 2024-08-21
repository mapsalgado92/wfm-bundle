from src.entities.data_classes import Shift
import jax.numpy as jnp
from typing import Optional


class RequirementShift(Shift):
    def __init__(
        self,
        id: str,
        start_time: float,
        duration: float,
        skill: str,
        daily_intervals: int = 24,
        coverage: Optional[jnp.ndarray] = None,
    ):
        super().__init__(id, start_time, duration, skill)
        self.start_int = jnp.trunc(self.start_time * daily_intervals / 24)
        self.duration = self.duration * daily_intervals / 24
        self.coverage = coverage

        if coverage is None:
            self.coverage = jnp.ones(self.duration)
        elif len(coverage) != self.duration:
            raise ValueError("Coverage must be of length duration")
        elif jnp.any((coverage > 1.0) & (coverage < 0.0)):
            raise ValueError("Coverage must have values between 0 and 1")
        else:
            self.coverage = coverage

    def get_shift_coverage(
        self, day_idx: int, daily_intervals: int, max_days: int
    ) -> jnp.ndarray:
        start_index = day_idx * daily_intervals + int(self.start_int)
        end_index = min(start_index + int(self.duration), daily_intervals * max_days)
        coverage = self.coverage[0 : end_index - start_index].copy()
        return (
            jnp.zeros(daily_intervals * max_days)
            .at[start_index:end_index]
            .set(coverage)
        )


class SchedulesShift(Shift):
    pass

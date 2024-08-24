from src.entities.data_classes import Shift
import jax.numpy as jnp
from typing import Optional, List


class RequirementShift(Shift):
    def __init__(
        self,
        shift: Shift,
        daily_intervals: int = 24,
        coverage: Optional[jnp.ndarray] = None,
        lower_bounds: Optional[List[int]] = None,
    ):
        self.id = shift.id
        self.skill = shift.skill
        self.start_time = shift.start_time
        self.start_int = jnp.trunc(self.start_time * daily_intervals / 24)
        self.duration = jnp.trunc(
            shift.duration * daily_intervals / 24
        )  # duration in intervals
        self.coverage = coverage
        self.lower_bounds = lower_bounds

        if coverage is None:
            self.coverage = jnp.ones(self.duration)
        elif len(coverage) != self.duration:
            raise ValueError("Coverage must be of length duration")
        elif jnp.any((coverage > 1.0) & (coverage < 0.0)):
            raise ValueError("Coverage must have values between 0 and 1")
        else:
            self.coverage = coverage

        if lower_bounds is None:
            self.lower_bounds = [0]
        elif jnp.min(lower_bounds) < 0:
            raise ValueError(
                "Lower bounds must be composed of ints greater or equal to 0"
            )
        else:
            self.lower_bounds = lower_bounds

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
    def __init__(self, shift: Shift, type: str = "work"):
        self.id = shift.id
        self.skill = shift.skill
        self.start_time = shift.start_time
        self.duration = shift.duration
        if type in ("work", "pto", "off"):
            self.type = type
        else:
            ValueError("Invalid shift type...")

    def is_pto(self):
        return self.type == "pto"

    def is_off(self):
        return self.type == "off"

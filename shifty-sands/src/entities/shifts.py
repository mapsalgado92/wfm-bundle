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
        self.task = shift.task
        self.start_time = shift.start_time
        self.start_int = int(self.start_time * daily_intervals / 24)
        self.duration = int(
            shift.duration * daily_intervals / 24
        )  # duration in intervals
        self.daily_intervals = daily_intervals
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
        self.task = shift.task
        self.start_time = shift.start_time
        self.duration = shift.duration
        self.reqs_shift: Optional[RequirementShift] = None
        if type in ("work", "pto", "off", "avail"):
            self.type = type
        else:
            ValueError("Invalid shift type...")

    def is_pto(self):
        return self.type == "pto"

    def is_off(self):
        return self.type == "off"

    def is_avail(self):
        return self.type == "off"

    # alternative initializers
    @staticmethod
    def from_reqs_shift(reqs_shift: RequirementShift, type: str) -> "SchedulesShift":
        new_shift = SchedulesShift(reqs_shift, type)
        new_shift.duration = new_shift.duration * 24 / reqs_shift.daily_intervals
        new_shift.reqs_shift = reqs_shift
        return new_shift

    @staticmethod
    def new_all_day_shift(id: str, task: str, type: str) -> "SchedulesShift":
        return SchedulesShift(
            Shift(id=id, start_time=0, duration=24.0, task=task), type=type
        )

    @staticmethod
    def new_work_shift(
        id: str, start_time: float, duration: float, task: str
    ) -> "SchedulesShift":
        return SchedulesShift(
            Shift(id=id, start_time=start_time, duration=duration, task=task),
            type="work",
        )

    @staticmethod
    def new_off_shift() -> "SchedulesShift":
        return SchedulesShift(
            Shift(id="off", start_time=0, duration=24.0, task="off"), type="off"
        )

    @staticmethod
    def new_available_shift() -> "SchedulesShift":
        return SchedulesShift(
            Shift(id="available", start_time=0, duration=24.0, task="available"),
            type="avail",
        )

    @staticmethod
    def new_unavailable_shift(start_time: float, end_time: float) -> "SchedulesShift":
        return SchedulesShift(
            Shift(id="unavailable", start_time=0, duration=24.0, task="unavailable"),
            "avail",
        )


class WeeklyShiftSequence:
    def __init__(self, num_weeks: int, default_shift: Shift):
        self.num_weeks: int = num_weeks
        self.default_shift: Shift = default_shift
        self.shifts: List[Shift] = [default_shift for _ in range(num_weeks * 7)]

    def get_weekly_shifts(self, week: int) -> List[Shift]:
        """
        Get sub-list of shifts for a given week.
        Weeks are index like, starting at 0 and ending in self.num_weeks
        """
        self.check_week(week)
        return self.shifts[week * 7 : (week + 1) * 7]

    def get_weekly_shift_coverage(
        self, week: int, shift_ids: Optional[List[str]] = None
    ) -> List[bool]:
        """
        Get a weekly list of bools. True when weekly shift id is in 'shift_ids'
        Weeks are index like, starting at 0 and ending in self.num_weeks
        """
        self.check_week(week)
        return [
            (s.id in shift_ids if shift_ids else True)
            for s in self.get_weekly_shifts(week)
        ]

    def set_weekly_shifts(self, shifts: List[Shift], week: int) -> None:
        self.check_week(week)
        self.shifts[week * 7 : (week + 1) * 7] = shifts

    def set_daily_shift(self, shift: Shift, day: int) -> None:
        """
        Day is an index, being 0 the first day and the last self.num_weeks * 7 - 1"
        """
        self.check_week(int(day / 7))
        self.shifts[day] = shift

    # Common check
    def check_week(self, week: int) -> None:
        if week >= self.num_weeks:
            raise ValueError(
                "Week is out of bounds, week can be in the range [0,%d)", self.num_weeks
            )

    @staticmethod
    def from_list(
        shift_list: List[Optional[Shift]], default_shift: Shift
    ) -> "WeeklyShiftSequence":
        num_weeks = len(shift_list) / 7
        if round(num_weeks) != num_weeks:
            raise ValueError(
                "Invalid length, shift_list must have a length divisible by 7"
            )
        wss = WeeklyShiftSequence(int(num_weeks), default_shift)

        for idx, s in enumerate(shift_list):
            if s != None:
                wss.set_daily_shift(s, idx)
            else:
                pass

        return wss

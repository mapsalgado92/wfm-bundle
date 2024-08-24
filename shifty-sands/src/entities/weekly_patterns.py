import jax.numpy as jnp
from src.entities.data_classes import Pattern
from src.entities.shifts import SchedulesShift
from typing import List


class WeeklyPattern:
    def __init__(self, pattern: Pattern) -> None:
        self.code: str = pattern.code
        self.pattern: List[bool] = [
            pattern.mon,
            pattern.tue,
            pattern.wed,
            pattern.thu,
            pattern.fri,
            pattern.sat,
            pattern.sun,
        ]
        self.opening: int = self.pattern.index(
            False
        )  # Number of consecutive working shifts at pattern start
        self.closing: int = (
            len(pattern) - pattern[::-1].index(False) - 1
        )  # Number of consecutive working shifts at pattern en

    def blocks(self, gap: int = 1) -> int:
        blocks = 0
        consec_off = gap
        for p in self.pattern:
            if p:
                if consec_off >= gap:
                    blocks += 1
                consec_off = 0
            else:
                consec_off += 1
        return blocks

    @property
    def p_array(self) -> jnp.ndarray:
        return jnp.array([1 if p else 0 for p in self.pattern])


class ScheduesPattern(WeeklyPattern):
    def __init__(self, pattern: Pattern) -> None:
        super().__init__(pattern)
        self.schedules: List[SchedulesShift]

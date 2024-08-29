from src.entities.shifts import SchedulesShift
from typing import List, Tuple, Optional


class WeeklyPattern:
    def __init__(self, id: str, pattern: List[bool]) -> None:
        self.id = id
        self.pattern = pattern
        self.opening = pattern.index(
            False
        )  # Number of consecutive working shifts at pattern start
        self.closing: int = pattern[::-1].index(
            False
        )  # Number of consecutive working shifts at pattern en

    def blocks(self, gap: int = 1) -> Tuple[int, List[int]]:
        blocks = 0
        block_edges = []
        consec_off = gap
        for idx, is_work in enumerate(self.pattern):
            if is_work:
                if consec_off >= gap:
                    blocks += 1
                    block_edges.append(idx - 1)
                consec_off = 0
            else:
                consec_off += 1
        return (blocks, block_edges[1:])


class SchedulesWeeklyPattern(WeeklyPattern):
    def __init__(
        self, id: str, pattern: List[bool], shift_list: List[SchedulesShift]
    ) -> None:
        super().__init__(
            id,
            pattern,
        )
        self.shift_list = shift_list

    def get_shift_at(self, day: int) -> SchedulesShift:
        return self.shift_list[day]

    def get_shift_coverage(self, shift: SchedulesShift) -> List[bool]:
        return [shift.id == sched_shift.id for sched_shift in self.shift_list]

    @property
    def num_work_shifts(self) -> int:
        return sum([shift.is_work() for shift in self.shift_list])

    @property
    def earliest_start(self) -> Optional[float]:
        if self.num_work_shifts > 0:
            oob = 100.0  # guaranteed out of bounds value
            return min(
                shift.start_time if shift.is_work() else oob
                for shift in self.shift_list
            )
        else:
            return None

    @property
    def latest_start(self) -> Optional[float]:
        if self.num_work_shifts > 0:
            oob = -1.0  # guaranteed out of bounds value
            return max(
                shift.start_time if shift.is_work() else oob
                for shift in self.shift_list
            )
        else:
            return None

    def __repr__(self) -> str:
        sl_str = [shift.id for shift in self.shift_list]
        return f"SchedulesWeeklyPattern({sl_str})"

from src.entities.shifts import SchedulesShift
from typing import List, Tuple


class WeeklyPattern:
    def __init__(self, id: str, pattern: List[bool]) -> None:
        self.id = id
        self.pattern = pattern
        self.opening = pattern.index(
            False
        )  # Number of consecutive working shifts at pattern start
        self.closing: int = (
            len(pattern) - pattern[::-1].index(False) - 1
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
        start_times = [shift.start_time for shift in shift_list]
        self.earliest_start = min(start_times)
        self.latest_start = max(start_times)

    def get_shift_at(self, day: int):
        return self.shift_list[day]

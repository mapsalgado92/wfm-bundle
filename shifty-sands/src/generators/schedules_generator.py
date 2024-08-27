from src.entities.agents import SchedulesAgent
from src.generators.generator import Generator
from src.algorithms.params import Params
from src.entities.shifts import (
    RequirementShift,
    SchedulesShift,
    WeeklyShiftSequence,
)
from src.entities.weekly_patterns import WeeklyPattern, SchedulesWeeklyPattern

import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional
import random


class SchedulesRequirement:
    shifts: Tuple[RequirementShift]
    values: jnp.ndarray

    def get_weekly_values(self, week):
        return self.values[:, (week - 1) * 7 : week * 7]

    def get_weekly_abs_residuals(self, schedule_coverage: jnp.ndarray, week: int):
        weekly_values = self.get_weekly_values(week)
        if schedule_coverage.shape != weekly_values.shape:
            raise ValueError(
                "Weekly schedule coverage must have same shape as weekly values: %s",
                weekly_values.shape,
            )
        return schedule_coverage - self.get_weekly_values(week)

    def get_weekly_rel_residuals(
        self, schedule_coverage: jnp.ndarray, week: int, safety_factor: int = 100
    ):
        """
        Creates special Relative Residuals between coverage and requirements.
        Note: Safety factor must be bigger than 0 and prevents division by 0.
        Higher values reduce distortion of residuals.
        """
        if safety_factor < 0:
            raise ValueError("e must be greater than 0")
        return self.get_weekly_abs_residuals(schedule_coverage, week) / (
            self.get_weekly_values(week) + (1 / safety_factor)
        )


class Schedules:
    agent: SchedulesAgent
    scheduled_shifts: WeeklyShiftSequence
    availability: WeeklyShiftSequence

    def __init__(self, agent: SchedulesAgent, availability: WeeklyShiftSequence):
        self.agent = agent
        self.availability = availability
        self.scheduled_shifts = WeeklyShiftSequence.from_list(
            availability.shifts, availability.default_shift
        )

    def get_shift_coverage(week, shift_id) -> jnp.ndarray:
        return jnp.array(["..."])

    def get_hourly_coverage(week, shift_id) -> jnp.ndarray:
        return jnp.array(["..."])


class SchedGeneratorParams(Params):
    def __init__(
        self,
        passes: int = 3,
        max_consecutive_working_days: int = 6,
        min_consecutive_working_days: int = 2,
        days_off_to_break_consistency: int = 1,
        single_day_off_start_time_max_variance: float = 1.0,
        multi_day_off_start_time_max_variance: float = 12.0,
    ):
        super().__init__(type="sched_generator")
        self.passes = passes
        self.max_consecutive_working_days: int = max_consecutive_working_days
        self.min_consecutive_working_days: int = min_consecutive_working_days
        self.days_off_to_break_consistency: int = days_off_to_break_consistency
        self.single_day_off_start_time_max_variance = (
            single_day_off_start_time_max_variance
        )
        self.multi_day_off_start_time_max_variance = (
            multi_day_off_start_time_max_variance
        )


class SchedulesGenerator(Generator):
    def __init__(
        self,
        requirements: SchedulesRequirement,
        schedules: List[Schedules],
        params: SchedGeneratorParams = SchedGeneratorParams(),
    ):
        self.requirements: SchedulesRequirement = requirements
        self.schedules: List[Schedules] = schedules
        self.params: SchedGeneratorParams = params
        self.work_shifts: List[SchedulesShift] = (
            self._generate_schedules_shifts_from_reqs()
        )
        self.weekly_patterns: List[WeeklyPattern] = self._generate_base_weekly_pattens()

    def generate(self, num_weeks: int) -> List[Schedules]:
        self._validate_num_weeks(num_weeks)
        possible_patterns = self._generate_possible_patterns()
        randomized_order = list(range(self.num_schedules))
        for week in range(num_weeks):
            for pass_idx in range(self.params.passes):
                random.shuffle(randomized_order)  # actually randomize order
                is_last_pass = pass_idx == self.params.passes - 1
                for sched_idx in randomized_order:
                    self._generate_weekly_schedules(week, sched_idx, possible_patterns)
        return self.schedules

    def _generate_base_weekly_pattens(self) -> List[WeeklyPattern]:
        return [
            WeeklyPattern(id=f"p{idx}", pattern=p)
            for idx, p in enumerate(
                [
                    [True, True, True, True, True, False, False],
                    [True, True, True, True, False, False, True],
                    [True, True, True, False, False, True, True],
                    [True, True, False, False, True, True, True],
                    [True, False, False, True, True, True, True],
                    [False, False, True, True, True, True, True],
                    [False, True, True, True, True, True, False],
                    [True, False, True, True, True, True, False],
                    [True, True, False, True, True, True, False],
                    [True, True, True, False, True, True, False],
                    [True, True, False, True, True, False, True],
                    [True, False, True, True, True, False, True],
                    [False, True, True, True, True, False, True],
                    [False, True, True, True, False, True, True],
                    [False, True, True, False, True, True, True],
                    [True, False, True, True, False, True, True],
                ]
            )
        ]

    def _generate_schedules_shifts_from_reqs(self) -> List[SchedulesShift]:
        return [
            SchedulesShift.from_reqs_shift(req_shift, type="work")
            for req_shift in self.requirements.shifts
        ]

    def _generate_possible_patterns(self) -> List[SchedulesWeeklyPattern]:
        off_shift = SchedulesShift.new_off_shift()

        blocks_gap = self.params.days_off_to_break_consistency
        single_off_start_var = self.params.single_day_off_start_time_max_variance
        multi_off_start_var = self.params.multi_day_off_start_time_max_variance

        shifts = self.work_shifts
        weekly_patterns = self.weekly_patterns
        possible_patterns = []

        for weekly_pattern in weekly_patterns:
            pattern = weekly_pattern.pattern
            blocks, edges = weekly_pattern.blocks(blocks_gap)
            if blocks == 1:
                for shift in shifts:
                    compound_id = f"{weekly_pattern.id}_{shift.id}"
                    possible_patterns.append(
                        SchedulesWeeklyPattern(
                            id=compound_id,
                            pattern=pattern,
                            shift_list=[
                                shift if is_work else off_shift for is_work in pattern
                            ],
                        )
                    )
            else:
                for edge in edges:
                    consec_off = list.index(pattern[edge:], True)
                    for first_shift in shifts:
                        for second_shift in shifts:
                            diff_start = abs(
                                second_shift.start_time - first_shift.start_time
                            )

                            valid_diff = (
                                (abs(diff_start) > single_off_start_var)
                                if consec_off == 1
                                else (abs(diff_start) > multi_off_start_var)
                            )

                            if valid_diff:
                                first_section = [
                                    first_shift if is_work else off_shift
                                    for is_work in pattern[0:edge]
                                ]

                                second_section = [
                                    second_shift if is_work else off_shift
                                    for is_work in pattern[edge:]
                                ]

                                compound_id = f"{weekly_pattern.id}_{first_shift.id}_{second_shift.id}"

                                possible_patterns.append(
                                    SchedulesWeeklyPattern(
                                        id=compound_id,
                                        pattern=pattern,
                                        shift_list=[*first_section, *second_section],
                                    )
                                )
        return possible_patterns

    #####################################------------------------

    def _generate_weekly_schedules(
        self,
        week: int,
        sched_idx: int,
        possible_patterns: List[SchedulesWeeklyPattern],
    ) -> None:

        availability = self.schedules[sched_idx].availability
        scheduled_shifts = self.schedules[sched_idx].scheduled_shifts
        agent = self.schedules[sched_idx].agent

        off_cov = jnp.array(
            availability.get_weekly_shift_coverage(week, ["off"])
        ).astype(int)
        available_cov = jnp.array(
            availability.get_weekly_shift_coverage(week, ["available"])
        ).astype(int)
        shift_cov = (
            jnp.array(availability.get_weekly_shift_coverage(week)).astype(int)
            - off_cov
            - available_cov
        )

        # LOOK AHEAD TODO: This logic misses cases where we have forced working shifts
        opening_next_week = None
        if week + 1 < availability.num_weeks:
            next_week_off_cov = availability.get_weekly_shift_coverage(
                week + 1, ["off"]
            )
            number_of_no_next_week = sum(next_week_off_cov)
            if number_of_no_next_week > 1:
                opening_next_week = next_week_off_cov.index(True)

        valid_poss = []

        for pattern in possible_patterns:
            if not self._valid_consecutive_working_days(
                agent.consec_work_carry + pattern.opening
            ):
                continue
            if opening_next_week:
                if not self._valid_consecutive_working_days(
                    pattern.closing + opening_next_week
                ):
                    continue
            if not self._valid_agent_start_times(agent, pattern):
                continue

            valid_poss.append(pattern)

        return None

    #####################################------------------------
    # validations
    def _valid_consecutive_working_days(self, consec_working_days: 6) -> bool:
        return (
            self.params.min_consecutive_working_days
            <= consec_working_days
            <= self.params.max_consecutive_working_days
        )

    def _valid_agent_start_times(
        self, agent: SchedulesAgent, pattern: SchedulesWeeklyPattern
    ) -> bool:
        return (pattern.earliest_start >= agent.constarints.early_start) and (
            pattern.latest_start <= agent.constarints.late_start
        )

    def _valid_shifts(
        self, off_cov, shift_cov, pattern: SchedulesWeeklyPattern
    ) -> bool:
        # Check off shifts
        if any(
            [
                (is_off == 1 and not pattern.get_shift_at(idx).is_off())
                for idx, is_off in enumerate(off_cov)
            ]
        ):
            return False

        # Check work/pto shifts
        elif any(
            [
                (is_not_off == 1 and pattern.get_shift_at(idx).is_off())
                for idx, is_not_off in enumerate(shift_cov)
            ]
        ):
            return False

        else:
            return True

    def _validate_num_weeks(self, num_weeks: int) -> bool:
        return all([self.requirements.values.shape[1] >= num_weeks * 7])

    @staticmethod
    def setup_schedules(
        agents: List[SchedulesAgent],
        availability_dict: Dict[str, List[Optional[SchedulesShift]]],
    ) -> List[Schedules]:
        return [
            Schedules(
                agent=agent,
                availability=WeeklyShiftSequence.from_list(
                    availability_dict.get(
                        agent.id,
                        [None, None, None, None, None, None, None],
                    ),
                    default_shift=SchedulesShift.new_available_shift(),
                ),
            )
            for agent in agents
        ]

    @property
    def num_schedules(self) -> int:
        return len(self.schedules)

    @property
    def num_shifts(self) -> int:
        return len(self.requirements.shifts)

from src.entities.agents import SchedulesAgent
from src.generators.generator import Generator
from src.algorithms.params import Params
from src.entities.shifts import (
    SchedulesShift,
    WeeklyShiftSequence,
)
from src.entities.weekly_patterns import WeeklyPattern, SchedulesWeeklyPattern
from src.algorithms.error_functions import (
    mse_with_asym_neg_penalty,
    mre_with_asym_neg_penalty,
)

import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional
import random

from functools import lru_cache
import logging


class SchedulesRequirement:
    shifts: Tuple[SchedulesShift]
    values: jnp.ndarray

    def __init__(self, shifts: Tuple[SchedulesShift], values: jnp.ndarray) -> None:
        if values.shape[0] != len(shifts):
            raise ValueError(
                "Values and shifts are not compatible, values.shape[0] must match number of shifts"
            )
        else:
            self.shifts = shifts
            self.values = values

    def get_weekly_values(self, week: int):
        """
        Note that week is an index.
        """
        return self.values[:, (week) * 7 : (week + 1) * 7]

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

    def get_multi_shift_weekly_coverage(
        self, week, shifts: Tuple[SchedulesShift], from_avail: bool = False
    ) -> jnp.ndarray:
        coverage = []
        shift_sequence = self.scheduled_shifts if not from_avail else self.availability

        for shift in shifts:
            coverage.append(shift_sequence.get_weekly_shift_coverage(week, [shift.id]))

        return jnp.array(coverage).astype(int)


class SchedGeneratorParams(Params):
    def __init__(
        self,
        passes: int = 3,
        max_consecutive_working_days: int = 6,
        min_consecutive_working_days: int = 2,
        days_off_to_break_consistency: int = 1,
        single_day_off_start_time_max_variance: float = 1.0,
        multi_day_off_start_time_max_variance: float = 12.0,
        error_asym_factor: float = 0.8,
        error_type: str = "abs",  # can be "abs" or "rel"
        rel_error_safety_factor: Optional[int] = None,  # only erquired for rel error
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
        self.error_asym_factor = error_asym_factor
        self.error_type = error_type
        self.rel_error_safety_factor = rel_error_safety_factor
        # Default Only
        self.oc_shift = SchedulesShift.new_all_day_shift(
            id="oc", task="oc", type="avail"
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
        possible_patterns = self.generate_possible_patterns()
        randomized_order = list(range(self.num_schedules))
        for week in range(num_weeks):
            logging.info("[WEEK] Generating week %d", week)
            for pass_idx in range(self.params.passes):
                logging.info("[PASS] Generating pass %d", pass_idx)
                random.shuffle(randomized_order)  # actually randomize order
                is_last_pass = pass_idx == self.params.passes - 1
                for sched_idx in randomized_order:
                    logging.info("[SCHED] Generating schedules %d", sched_idx)
                    generation_schedules = self.schedules[sched_idx]
                    logging.info("[SCHED] -> Resetting Schedules")
                    generation_schedules.scheduled_shifts.reset_weekly_shifts(week)
                    logging.info("[SCHED] -> Generating Valid Patterns")
                    valid_patterns = self._generate_valid_weekly_patterns(
                        week, sched_idx, possible_patterns
                    )
                    logging.info("[SCHED] --- valid patterns [%d]", len(valid_patterns))

                    if len(valid_patterns) > 0:
                        logging.info("[SCHED] -> Calculating valid pattern errors")
                        weekly_coverage = self.get_full_schedules_weekly_coverage(week)
                        pattern_costs = [
                            dict(
                                cost=self.cost_function(week, weekly_coverage, pattern),
                                shift_list=pattern.shift_list,
                            )
                            for pattern in valid_patterns
                        ]

                        best_pattern = min(pattern_costs, key=lambda x: x["cost"])
                        logging.info(
                            "[SCHED] --- best pattern cost [%d]", best_pattern["cost"]
                        )

                        logging.info("[SCHED] Adding best pattern to schedules")
                        generation_schedules.scheduled_shifts.set_weekly_shifts(
                            min(pattern_costs, key=lambda x: x["cost"])["shift_list"],
                            week,
                        )
                    else:
                        logging.info(
                            "[SCHED] -> No valid patterns, adding Overconstrained week"
                        )
                        avail_shift_list = (
                            generation_schedules.availability.get_weekly_shifts(week)
                        )
                        ooc_shift_list = [
                            self.params.oc_shift if shift.is_avail() else shift
                            for shift in avail_shift_list
                        ]
                        generation_schedules.scheduled_shifts.set_weekly_shifts(
                            ooc_shift_list, week
                        )
                    if is_last_pass:
                        logging.info("[SCHED] Last Pass, updating agent state")
                        updated_agent = self.update_agent_state(
                            generation_schedules.agent,
                            generation_schedules.scheduled_shifts.get_weekly_shifts(
                                week
                            ),
                        )
                        logging.info(
                            "[SCHED] -> Updated state with carry: %s",
                            updated_agent.consec_work_carry,
                        )
                        logging.info(
                            "[SCHED] -> Updated state with start_time: %s",
                            updated_agent.consistency_start_time,
                        )

        return self.schedules

    # Exposed methods
    def generate_possible_patterns(self) -> Tuple[SchedulesWeeklyPattern]:
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
                                (abs(diff_start) <= single_off_start_var)
                                if consec_off == 1
                                else (abs(diff_start) <= multi_off_start_var)
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
        return tuple(possible_patterns)

    @lru_cache(maxsize=None)
    def get_pattern_coverage(self, pattern: SchedulesWeeklyPattern) -> jnp.ndarray:
        return jnp.array(
            [pattern.get_shift_coverage(shift) for shift in self.requirement_shifts]
        ).astype(int)

    def get_full_schedules_weekly_coverage(self, week: int):

        return sum(
            [
                schedule.get_multi_shift_weekly_coverage(week, self.requirement_shifts)
                for schedule in self.schedules
            ]
        )

    def update_agent_state(
        self, agent: SchedulesAgent, shifts: List[SchedulesShift]
    ) -> SchedulesAgent:
        off_cov = [shift.is_off() for shift in shifts]
        if off_cov[-1]:
            # last day is off
            if off_cov[-2]:
                # second to last day is also off
                carry = -1
                start_time = shifts[-3].start_time if shifts[-3].is_work() else None
            else:
                carry = 0
                start_time = shifts[-2].start_time if shifts[-2].is_work() else None
        else:
            carry = off_cov[::-1].index(True) if True in off_cov else -2
            start_time = shifts[-1].start_time if shifts[-1].is_work() else None

        agent.consec_work_carry = carry
        agent.consistency_start_time = start_time
        return agent

    def cost_function(
        self,
        week: int,
        weekly_coverage: jnp.ndarray,
        pattern: SchedulesWeeklyPattern,
    ) -> float:
        values = weekly_coverage + self.get_pattern_coverage(pattern)
        target = self.requirements.get_weekly_values(week)
        if self.params.error_type == "abs":
            return mse_with_asym_neg_penalty(
                values, target, self.params.error_asym_factor
            ).item()
        else:
            return mre_with_asym_neg_penalty(
                values,
                target,
                self.params.error_asym_factor,
                self.params.rel_error_safety_factor,
            ).item()

    # Internal methods
    def _generate_base_weekly_pattens(self) -> Tuple[WeeklyPattern]:
        return tuple(
            [
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
        )

    def _generate_schedules_shifts_from_reqs(self) -> Tuple[SchedulesShift]:
        if isinstance(self.requirements.shifts[0], SchedulesShift):
            return self.requirements.shifts
        else:
            return tuple(
                [
                    SchedulesShift.from_reqs_shift(req_shift, type="work")
                    for req_shift in self.requirements.shifts
                ]
            )

    def _reset_weekly_shifts(self, week: int, sched_idx: int) -> None:
        self.schedules[sched_idx].scheduled_shifts.reset_weekly_shifts(week)

    @lru_cache(maxsize=None)  # caching result to save computations
    def _generate_valid_weekly_patterns(
        self,
        week: int,
        sched_idx: int,
        possible_patterns: Tuple[SchedulesWeeklyPattern],
    ) -> Tuple[SchedulesWeeklyPattern]:
        availability = self.schedules[sched_idx].availability
        agent = self.schedules[sched_idx].agent

        off_cov = jnp.array(
            availability.get_weekly_shift_coverage(week, ["off"])
        ).astype(int)
        avail_cov = jnp.array(
            availability.get_weekly_shift_coverage(week, ["available", "unavaliable"])
        ).astype(int)
        shift_cov = (
            jnp.array(availability.get_weekly_shift_coverage(week)).astype(int)
            - off_cov
            - avail_cov
        )
        shift_avail = availability.get_weekly_shifts(week)

        # LOOK AHEAD TODO: This logic misses cases where we have forced working shifts
        min_opening_next_week = None
        start_time_next_week = None
        if week + 1 < availability.num_weeks:
            off_cov_next_week = jnp.array(
                availability.get_weekly_shift_coverage(week + 1, ["off"])
            ).astype(int)
            avail_cov_next_week = jnp.array(
                availability.get_weekly_shift_coverage(
                    week + 1, ["available", "unavaliable"]
                )
            ).astype(int)
            shift_cov_next_week = (
                jnp.array(availability.get_weekly_shift_coverage(week + 1)).astype(int)
                - off_cov_next_week
                - avail_cov_next_week
            )
            shift_avail_next_week = availability.get_weekly_shifts(week + 1)

            number_of_no_next_week = off_cov_next_week.sum().item()
            number_of_shifts_next_week = shift_cov_next_week.sum().item()
            # Look ahead opening
            if number_of_no_next_week > 1:
                min_opening_next_week = jnp.argmax(off_cov_next_week == 1).item()
            elif number_of_shifts_next_week > 0:
                # check how many consecutive non-off shifts
                last_opening_shift_index = jnp.argmax(shift_cov_next_week == 0).item()
                min_opening_next_week = (
                    max(
                        last_opening_shift_index, 2
                    )  # for safety consider min 2 consec days
                    if last_opening_shift_index > 0
                    else None
                )
            # Look ahead start times
            if shift_avail_next_week[0].is_work() or shift_avail_next_week[1].is_work():
                start_time_next_week = (
                    shift_avail_next_week[0].start_time
                    if shift_avail_next_week[0].is_work()
                    else shift_avail_next_week[1].start_time
                )

        valid_poss = []

        for pattern in possible_patterns:
            if not self._valid_consecutive_working_days(
                max(agent.consec_work_carry, 0) + pattern.opening
            ):
                continue
            if min_opening_next_week and not self._valid_consecutive_working_days(
                pattern.closing + min_opening_next_week
            ):
                continue
            if start_time_next_week and not self._valid_next_week_start_times(
                pattern, start_time_next_week, min_opening_next_week
            ):
                continue
            if not self._valid_agent_start_times(agent, pattern):
                continue

            if not self._valid_off_pattern(off_cov, shift_cov, pattern):
                continue

            if not self._valid_consistency(agent, pattern):
                continue

            if not self._valid_availability_consistency(shift_avail, pattern):
                continue

            valid_poss.append(self._get_pattern_with_availability(pattern, shift_avail))

        return tuple(valid_poss)

    def _valid_consecutive_working_days(self, consec_working_days: int) -> bool:
        if consec_working_days == 0:
            return True
        else:
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

    def _valid_next_week_start_times(
        self,
        pattern: SchedulesWeeklyPattern,
        start_time_next_week: float,
        has_opening: bool,
    ):
        if pattern.closing == 0:
            if pattern.get_shift_at(5).is_off():
                return (
                    abs(start_time_next_week - pattern.get_shift_at(4).start_time)
                    <= self.params.multi_day_off_start_time_max_variance
                )
            elif has_opening:
                return (
                    abs(start_time_next_week - pattern.get_shift_at(5).start_time)
                    <= self.params.single_day_off_start_time_max_variance
                )
            else:
                return (
                    abs(start_time_next_week - pattern.get_shift_at(5).start_time)
                    <= self.params.multi_day_off_start_time_max_variance
                )
        else:
            if has_opening:
                return start_time_next_week == pattern.get_shift_at(6)
            else:
                return (
                    abs(start_time_next_week - pattern.get_shift_at(5).start_time)
                    <= self.params.single_day_off_start_time_max_variance
                )

    def _valid_off_pattern(
        self, off_cov: List[int], shift_cov: List[int], pattern: SchedulesWeeklyPattern
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

    def _valid_consistency(
        self,
        agent: SchedulesAgent,
        pattern: SchedulesWeeklyPattern,
    ) -> bool:
        carry = agent.consec_work_carry
        pattern_consist_start = pattern.get_shift_at(0).start_time
        agent_consist_start = (
            agent.consistency_start_time
            if agent.consistency_start_time != None
            else pattern_consist_start
        )
        params = self.params
        if carry < -1:
            return True
        elif carry == -1:
            return (
                abs(pattern_consist_start - agent_consist_start)
                <= params.multi_day_off_start_time_max_variance
            )
        elif carry == 0:
            return (
                abs(pattern_consist_start - agent_consist_start)
                <= params.single_day_off_start_time_max_variance
            )
        else:
            return agent_consist_start == pattern_consist_start

    def _valid_availability_consistency(
        self, shift_avail: List[SchedulesShift], pattern: SchedulesWeeklyPattern
    ) -> bool:
        for day in range(7):
            avail_shift = shift_avail[day]
            pattern_shift = pattern.get_shift_at(day)
            # Check work day availability (keep starting time)
            if (
                avail_shift.is_work()
                and avail_shift.start_time != pattern_shift.start_time
            ):
                return False
            # Check unavailability start range
            elif avail_shift.id == "unavailable" and (
                avail_shift.start_time
                < pattern_shift.start_time
                < avail_shift.start_time + avail_shift.duration
            ):
                return False
            # Check availability start range
            elif avail_shift.id == "available" and not (
                avail_shift.start_time
                <= pattern_shift.start_time
                <= avail_shift.start_time + avail_shift.duration
            ):
                return False

        return True

    def _validate_num_weeks(self, num_weeks: int) -> bool:
        return all([self.requirements.values.shape[1] >= num_weeks * 7])

    def _get_pattern_with_availability(
        self, pattern: SchedulesWeeklyPattern, availability_shifts: List[SchedulesShift]
    ) -> SchedulesWeeklyPattern:
        new_shift_list = []
        for idx, shift in enumerate(availability_shifts):
            if shift.is_avail():
                new_shift_list.append(pattern.shift_list[idx])
            else:
                new_shift_list.append(shift)
        return SchedulesWeeklyPattern(pattern.id, pattern.pattern, new_shift_list)

    def _special_check(
        self, schedules: Schedules, week, pattern: SchedulesWeeklyPattern
    ) -> bool:
        if week > 0:
            special_check_prev_week = schedules.scheduled_shifts.get_weekly_shifts(
                week - 1
            )
            if special_check_prev_week[-1].is_off() or pattern.get_shift_at(0).is_off():
                return True
            if (
                special_check_prev_week[-1].start_time
                != pattern.get_shift_at(0).start_time
            ):
                # Check breakpoint
                return False
        else:
            return True

    @staticmethod
    def setup_schedules(
        agents: List[SchedulesAgent],
        availability_dict: Dict[str, List[Optional[SchedulesShift]]],
        num_weeks: int = 1,
    ) -> List[Schedules]:
        return [
            Schedules(
                agent=agent,
                availability=WeeklyShiftSequence.from_list(
                    availability_dict.get(
                        agent.id,
                        [None for _ in range(num_weeks * 7)],
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

    @property
    def requirement_shifts(self) -> Tuple[SchedulesShift]:
        return tuple(self.requirements.shifts)

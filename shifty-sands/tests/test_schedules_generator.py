from conftest import TestObjects
from src.generators.schedules_generator import (
    SchedulesGenerator,
    SchedulesRequirement,
    SchedulesAgent,
)

from src.entities.shifts import RequirementShift, SchedulesShift, WeeklyShiftSequence
from src.entities.data_classes import Shift

import jax.numpy as jnp

from collections import Counter
from typing import List

from pytest import fixture


@fixture()
def local_test_objects():
    local_objects = TestObjects()

    db_shifts = [
        Shift(id="A", start_time=9.0, duration=9.0, task="task1"),
        Shift(id="B", start_time=12.0, duration=9.0, task="task1"),
        Shift(id="C", start_time=16.0, duration=9.0, task="task1"),
        Shift(id="D", start_time=23.0, duration=9.0, task="task1"),
    ]

    req_shifts = [RequirementShift(s) for s in db_shifts]
    local_objects.add_object("req_shifts", req_shifts)

    sched_shifts = [SchedulesShift.from_reqs_shift(s, "work") for s in req_shifts]
    local_objects.add_object("sched_shifts", sched_shifts)

    # 2 weeks of requirement
    values = jnp.array(
        [
            [3, 2, 2, 2, 2, 1, 1, 3, 2, 2, 2, 2, 1, 1],
            [4, 3, 3, 3, 3, 2, 2, 4, 3, 3, 3, 3, 2, 2],
            [2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # unused
        ]
    )

    sched_reqs = SchedulesRequirement(tuple(req_shifts), values)

    local_objects.add_object("sched_reqs", sched_reqs)

    local_objects.add_object(
        "agents",
        [
            SchedulesAgent("A1", tasks=["task1", "task2"]),
            SchedulesAgent("A2", tasks=["task1", "task2"]),
            SchedulesAgent("A3", tasks=["task1", "task2"]),
            SchedulesAgent("A4", tasks=["task1", "task2"]),
            SchedulesAgent("A5", tasks=["task1", "task2"]),
            SchedulesAgent("A6", tasks=["task1"]),
            SchedulesAgent("A7", tasks=["task1"]),
            SchedulesAgent("A8", tasks=["task1"]),
        ],
    )

    vac = SchedulesShift.new_all_day_shift(id="vac", task="vac", type="pto")
    off = SchedulesShift.new_off_shift()
    unav_16 = SchedulesShift.new_unavailable_shift(0, 16)
    task2_9 = SchedulesShift.new_work_shift("Z", 9.0, 9.0, "task2")
    task1_12 = sched_shifts[1]

    local_objects.add_object(
        "availability_dict",
        {
            "A1": [None, None, unav_16, vac, vac, off, off],
            "A2": [task1_12, None, None, None, None, off, off],
            "A3": [off, off, None, None, None, None, None],
            "A5": [None, None, task2_9, off, None, None, None],
            "A8": [vac, vac, off, off, None, None, None],
        },
    )

    return local_objects


def test_weekly_shift_sequence_initializations():
    default_shift = SchedulesShift.new_available_shift()

    num_weeks = 3
    wss = WeeklyShiftSequence(num_weeks, default_shift)

    assert len(wss.shifts) == num_weeks * 7

    normal_shift = SchedulesShift.new_work_shift(
        id="task1-12", start_time=12.0, duration=8.0, task="task1"
    )

    shift_list = [normal_shift if b % 2 == 0 else None for b in range(7 * num_weeks)]

    wss = WeeklyShiftSequence.from_list(shift_list, default_shift)

    assert len(wss.shifts) == 7 * num_weeks

    counter = Counter([s.id for s in wss.shifts])

    assert counter.get("available") - counter.get("task1-12") == (
        0 if num_weeks % 2 == 0 else -1
    )


def test_schedules_generator_initialization(local_test_objects: TestObjects):
    sched_reqs: SchedulesRequirement = local_test_objects.get_object("sched_reqs")
    agents: List[SchedulesAgent] = local_test_objects.get_object("agents")
    availability: dict[str, list] = local_test_objects.get_object("availability_dict")

    schedules = SchedulesGenerator.setup_schedules(agents, availability)

    sched_gen = SchedulesGenerator(sched_reqs, schedules)

    assert [ws in sched_reqs.shifts for ws in sched_gen.work_shifts]
    # TODO:MORE ASSERTIONS


def test_generate_possible_patterns_output_is_correct(
    local_test_objects: TestObjects,
):
    sched_reqs = local_test_objects.get_object("sched_reqs")
    agents = local_test_objects.get_object("agents")
    availability = local_test_objects.get_object("availability_dict")
    schedules = SchedulesGenerator.setup_schedules(agents, availability)

    sched_gen = SchedulesGenerator(requirements=sched_reqs, schedules=schedules)

    sched_gen.params.days_off_to_break_consistency = 3
    possible_patterns = sched_gen.generate_possible_patterns()
    possible_patterns_first_run = len(possible_patterns)

    sched_gen.params.days_off_to_break_consistency = 2
    possible_patterns = sched_gen.generate_possible_patterns()
    possible_patterns_second_run = len(possible_patterns)

    sched_gen.params.days_off_to_break_consistency = 1
    possible_patterns = sched_gen.generate_possible_patterns()
    possible_patterns_third_run = len(possible_patterns)

    assert (
        possible_patterns_first_run
        == len(sched_gen.weekly_patterns) * sched_gen.num_shifts
    )

    assert possible_patterns_second_run > possible_patterns_first_run

    assert possible_patterns_third_run > possible_patterns_second_run


def test_generate_weekly_schedules(local_test_objects: TestObjects):
    sched_reqs = local_test_objects.get_object("sched_reqs")
    agents = local_test_objects.get_object("agents")
    availability = local_test_objects.get_object("availability_dict")
    schedules = SchedulesGenerator.setup_schedules(agents, availability)

    sched_gen = SchedulesGenerator(requirements=sched_reqs, schedules=schedules)
    sched_gen.generate(1)

    assert True


def test_possible_patterns_validations(local_test_objects: TestObjects):
    sched_reqs = local_test_objects.get_object("sched_reqs")
    agents = local_test_objects.get_object("agents")
    availability = local_test_objects.get_object("availability_dict")
    schedules = SchedulesGenerator.setup_schedules(agents, availability)

    sched_gen = SchedulesGenerator(requirements=sched_reqs, schedules=schedules)
    poss_paterns = sched_gen.generate_possible_patterns()

    # Agent with unavailable (0.0-16.0) foreced off and vacation (single pattern)
    valid_poss_A1 = sched_gen._generate_valid_weekly_patterns(
        week=0, sched_idx=0, possible_patterns=poss_paterns
    )

    # Agent with fixed days off, and a fixed working shift (one possible solution)
    valid_poss_A2 = sched_gen._generate_valid_weekly_patterns(
        week=0, sched_idx=1, possible_patterns=poss_paterns
    )

    # Agent with forced special task shift and forced off 9.0
    valid_poss_A5 = sched_gen._generate_valid_weekly_patterns(
        week=0, sched_idx=4, possible_patterns=poss_paterns
    )

    # Agent with no restrictions (only fact that carry is 0 excludes some possibilities)
    valid_poss_A7 = sched_gen._generate_valid_weekly_patterns(
        week=0, sched_idx=6, possible_patterns=poss_paterns
    )

    # Agent both forced off and vacation
    valid_poss_A8 = sched_gen._generate_valid_weekly_patterns(
        week=0, sched_idx=7, possible_patterns=poss_paterns
    )

    assert len(valid_poss_A1) == 2

    assert len(valid_poss_A2) == 1 and valid_poss_A2[0].earliest_start == 12.0

    assert len(valid_poss_A5) == 9

    assert len(valid_poss_A7) == sum([poss.opening != 1 for poss in poss_paterns])

    assert len(valid_poss_A8) == 12


def test_overconstrained_output_when_no_valid_patterns(
    local_test_objects: TestObjects,
) -> None:
    sched_reqs = local_test_objects.get_object("sched_reqs")
    agents = local_test_objects.get_object("agents")
    availability = local_test_objects.get_object("availability_dict")
    for key, value in availability.items():
        availability[key] = [*value, *[None, None, None, None, None, None, None]]
    schedules = SchedulesGenerator.setup_schedules(agents, availability, num_weeks=2)
    sched_gen = SchedulesGenerator(requirements=sched_reqs, schedules=schedules)
    sched_gen.generate(2)

    assert True

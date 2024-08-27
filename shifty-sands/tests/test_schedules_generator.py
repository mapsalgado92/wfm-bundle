from conftest import TestObjects
from src.generators.schedules_generator import (
    SchedulesGenerator,
    SchedulesRequirement,
    SchedulesAgent,
    SchedGeneratorParams,
    Schedules,
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
        Shift(id="A", start_time=9.0, duration=9.0, task="chat"),
        Shift(id="B", start_time=12.0, duration=9.0, task="chat"),
        Shift(id="C", start_time=16.0, duration=9.0, task="chat"),
        Shift(id="D", start_time=23.0, duration=9.0, task="chat"),
        Shift(id="E", start_time=9.0, duration=9.0, task="phone"),
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
            [2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 0],  # non-base
        ]
    )

    sched_reqs = SchedulesRequirement()
    sched_reqs.shifts = req_shifts  # Shifts Tuple
    sched_reqs.values = values  # Requirement Values
    local_objects.add_object("sched_reqs", sched_reqs)

    local_objects.add_object(
        "agents",
        [
            SchedulesAgent("A1", tasks=["task1", "task2"]),
            SchedulesAgent("A2", tasks=["task1", "task2"]),
            SchedulesAgent("A3", tasks=["task1", "task2"]),
            SchedulesAgent("A4", tasks=["task1", "task2"]),
            SchedulesAgent("A5", tasks=["task1"]),
            SchedulesAgent("A6", tasks=["task1"]),
            SchedulesAgent("A7", tasks=["task1"]),
            SchedulesAgent("A8", tasks=["task1"]),
        ],
    )

    vac = SchedulesShift.new_all_day_shift(id="vac", task="vac", type="pto")
    off = SchedulesShift.new_off_shift()

    local_objects.add_object(
        "availability_dict",
        {
            "A1": [None, None, None, vac, vac, off, off],
            "A2": [None, None, None, None, None, off, off],
            "A5": [None, None, None, off, None, None, None],
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
    possible_patterns = sched_gen._generate_possible_patterns()
    possible_patterns_first_run = len(possible_patterns)

    sched_gen.params.days_off_to_break_consistency = 2
    possible_patterns = sched_gen._generate_possible_patterns()
    possible_patterns_second_run = len(possible_patterns)

    sched_gen.params.days_off_to_break_consistency = 1
    possible_patterns = sched_gen._generate_possible_patterns()
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
    sched_gen.generate(num_weeks=1)

    assert True

from src.entities.agents import SchedulesAgent
from src.entities.weekly_patterns import ScheduesPattern
from src.entities.shifts import RequirementShift
import jax.numpy as jnp
from typing import List, Tuple


class SchedulingRequirement:
    shifts: Tuple[RequirementShift]
    values: jnp.ndarray


class Schedules:
    agent: SchedulesAgent
    patterns: List[ScheduesPattern]

    def get_shift_coverage(shift_id) -> jnp.ndarray:
        return jnp.array[...]


class SchedulesGenerator:
    schedules: List[Schedules]
    requirements: SchedulingRequirement

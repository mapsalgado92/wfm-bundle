from dataclasses import dataclass
from typing import List, Optional
import jax.numpy as jnp
from src.entities.availability import AgentAvailability, DailyAvailability


@dataclass
class Constraints:
    early_start: int = 0  # earliest daily start of shift [0-24]
    late_start: int = 24  # latest daily start of shift (excluding) [0-24]
    timezone_correction: int = 0  # hourly timezone correction (from UTC) [-12:+12]
    max_consec_shifts: int = 5  # number of maximum consecutive working days
    consistency_gap: int = 2  # min days off to break consistency
    consistency_hours: int = 0  # hours between shift starts to keep consistency

    def __post_init__(self):
        if self.early_start >= self.late_start:
            raise ValueError("early_start must be less than late_start")
        if not 0 <= self.max_consec_shifts <= 12:
            raise ValueError(
                "max_consec_shifts must be non-negative and smaller than 12"
            )
        if not self.consistency_gap in [1, 2]:
            raise ValueError("consistency_gap must be 1 or 2")
        if not 0 <= self.consistency_hours <= 6:
            raise ValueError("consistency_hours must be in interval [0-6]")


@dataclass
class Preferences:
    pass


class Agent:
    def __init__(
        self,
        id: str,
        skills: List[str],
        constraints: Constraints,
        # preferences: Preferences,
        availability: Optional[AgentAvailability],
    ) -> None:
        self.id = id
        skills: List[str] = skills
        self.constarints = constraints
        # self.preferences = {}
        self.availability = availability

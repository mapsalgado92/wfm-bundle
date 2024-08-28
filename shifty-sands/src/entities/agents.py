from dataclasses import dataclass
from typing import List, Any, Optional
import jax.numpy as jnp


@dataclass
class AgentConstraints:
    early_start: float = 0.0  # earliest daily start of shift [0-24]
    late_start: float = 24.0  # latest daily start of shift (excluding) [0-24]

    def __post_init__(self):
        if self.early_start >= self.late_start:
            raise ValueError("early_start must be less than late_start")


class SchedulesAgent:
    def __init__(
        self,
        id: str,
        tasks: List[str],
        constraints: AgentConstraints = AgentConstraints(),
        consec_work_carry: int = -1,
        consistency_start_time: float = 12.0,
    ) -> None:
        self.id = id
        self.tasks = tasks
        self.constarints = constraints
        self.consec_work_carry = consec_work_carry
        self.consistency_start_time = consistency_start_time

    def update_agent_carry(self, carry: int, start_time: float) -> None:
        self.consec_work_carry = carry
        self.consistency_start_time = start_time

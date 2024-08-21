from typing import List
import jax.numpy as jnp


class WeeklyPattern:
    def __init__(self, id: str, pattern: List[bool]) -> None:
        self.id = id
        self.pattern = pattern
        self.opening = pattern.index(
            False
        )  # Number of consecutive working shifts at pattern start
        self.closing = (
            len(pattern) - pattern[::-1].index(False) - 1
        )  # Number of consecutive working shifts at pattern en

    @property
    def p_array(self) -> jnp.ndarray:
        return jnp.array([1 if p else 0 for p in self.pattern])

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


class ScheduedPattern(WeeklyPattern):
    pass

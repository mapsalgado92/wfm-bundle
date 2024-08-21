import jax.numpy as jnp
from jax import random
from typing import Dict, List


def generate_interval_requirements(
    weeks=4, base_demand=10, cycle_amplitude=5, weekend_effect=0.8
) -> jnp.ndarray:
    daily_cycle = jnp.sin(jnp.linspace(0, 14 * jnp.pi, 7 * 24))
    weekend_effect = jnp.concatenate(
        [jnp.ones(24 * 5), jnp.ones(24 * 2) * weekend_effect]
    )  # Lower volume on weekends
    return jnp.tile(
        daily_cycle * cycle_amplitude
        + base_demand * weekend_effect
        + 0.5 * random.normal(key=random.key(0), shape=(24 * 7,)),
        weeks,
    )

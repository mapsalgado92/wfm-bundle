import jax.numpy as jnp
from jax import jit, grad, random
from .params import GradientDescentParams, StochasticRoundingParams
from typing import Callable, Tuple


def gradient_descent(
    initial_weights: jnp.ndarray,
    target: jnp.ndarray,
    params: GradientDescentParams,
    loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    lower_bounded: bool = False,
) -> Tuple[jnp.ndarray, list]:
    gradient_function = jit(grad(loss_func))
    evolution = []
    weights = initial_weights.copy()
    for i in range(params.num_iterations):
        gradient = gradient_function(weights, target)

        weights = jnp.maximum(
            weights - params.learning_rate * gradient,
            initial_weights if lower_bounded else 0,
        )

        if i % params.snapshot_length == 0:

            error = loss_func(weights, target)

            evolution.append(
                {
                    "iteration_id": i,
                    "error": float(error),
                    "values": weights.copy(),
                }
            )

            print(f"Iteration {i}: Error = {error}")

    return weights, evolution


def stochastic_rounding(
    initial_weights: jnp.ndarray,
    target: jnp.ndarray,
    params: StochasticRoundingParams,
    loss_func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    seed: int = 0,
) -> Tuple[jnp.ndarray, list]:

    key = random.PRNGKey(seed)
    weights = initial_weights.copy()
    evolution = []
    if params.section_size > len(weights):
        raise ValueError("Section size must be smaller than length of weights.")
    for section_start in range(0, len(weights), params.section_size):
        section_end = section_start + params.section_size
        best_round_error = jnp.inf
        section_weights = weights[section_start:section_end]
        print("Section:", (section_start, section_end))
        for _ in range(params.passes):
            key, subkey = random.split(key, 2)
            rounded_weights = weights.at[section_start:section_end].set(
                jnp.where(
                    random.uniform(
                        subkey,
                        section_weights.shape,
                    )
                    < section_weights - jnp.floor(section_weights),
                    jnp.ceil(section_weights),
                    jnp.floor(section_weights),
                )
            )
            new_round_error = loss_func(rounded_weights, target)

            if new_round_error < best_round_error:
                best_round_error = new_round_error
                best_rounded_weights = rounded_weights

        print("Best Round Error:", best_round_error)
        weights = best_rounded_weights
        evolution.append(
            {
                "iteration_id": (section_start, section_end),
                "error": float(new_round_error),
                "values": best_rounded_weights.copy(),
            }
        )
    return weights, evolution

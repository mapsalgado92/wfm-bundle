from src.algorithms.optimization import (
    gradient_descent,
    GradientDescentParams,
    StochasticRoundingParams,
)
import jax.numpy as jnp
import pytest

from src.algorithms.params import Params


def test_gradient_specific_params_initialization():
    gd_params = GradientDescentParams(num_iterations=100)
    sr_params = StochasticRoundingParams(section_size=20)

    assert gd_params.num_iterations == 100
    assert sr_params.section_size == 20
    assert all((isinstance(p, Params) for p in [gd_params, sr_params]))

    # def test_gradient_descent_basic_outputs():
    initial_weights = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target = jnp.array([5.0, 3.0, 2.0, 5.0, 6.0])
    error_func = lambda weights, target: jnp.sum(
        (weights + jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]) - target) ** 2
    )
    params = GradientDescentParams(
        learning_rate=0.01, num_iterations=1000, snapshot_length=50
    )
    weights, evo = gradient_descent(initial_weights, target, params, error_func, False)

    assert len(weights) == 5
    assert len(evo) == jnp.ceil(params.num_iterations / params.snapshot_length)
    assert set(evo[0].keys()) == set(["iteration_id", "error", "values"])
    assert (
        error_func(weights, target) < params.learning_rate * 10
    )  # Function known to coverge...


def test_gradient_descent_lower_bounds():
    initial_weights = jnp.array([0.0, 1.0, 0.0, 10.0, 0.0])
    target = jnp.array([5.0, -3.0, 2.0, 5.0, 6.0])
    error_func = lambda weights, target: jnp.sum((2 * weights - target) ** 2)
    params = GradientDescentParams()
    (weights, _) = gradient_descent(
        initial_weights, target, params, error_func, lower_bounded=True
    )

    assert len(weights) == 5
    assert weights[1] == 1.0  # because it's low-bounded, should be negative
    assert weights[3] == 10.0  # because it's low-bounded, should tend to 5.0


def test_gradient_stochastic_rounding():
    initial_weights = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target = jnp.array([5.0, 3.0, 2.0, 5.0, 6.0])
    error_func = lambda weights, target: jnp.sum(
        (weights + jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]) - target) ** 2
    )
    params = GradientDescentParams(
        learning_rate=0.01, num_iterations=1000, snapshot_length=50
    )
    weights, evo = gradient_descent(initial_weights, target, params, error_func, True)

    assert len(weights) == 5
    assert len(evo) == jnp.ceil(params.num_iterations / params.snapshot_length)
    assert set(evo[0].keys()) == set(["iteration_id", "error", "values"])
    assert (
        error_func(weights, target) < params.learning_rate * 10
    )  # Function known to coverge...

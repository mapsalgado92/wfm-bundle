from src.shift_requirements import (
    ShiftRequirements,
    Params,
    GradientDescentParams,
    StochasticRoudningParams,
    ShiftBlueprint,
)
from conftest import TestObjects
import jax.numpy as jnp
import pytest


@pytest.fixture
def local_tests_objects(test_objects) -> TestObjects:
    shifts = [
        ShiftBlueprint(8, 9, "Morning"),
        ShiftBlueprint(14, 9, "Afternoon"),
        ShiftBlueprint(23, 9, "Night"),
    ]
    test_objects.add_object("shifts", shifts)

    return test_objects


def test_shift_blueprint_inicialises_properly():
    # Successful Initialization without coverage
    sb = ShiftBlueprint(2, 9, "Morning")
    assert sb.duration == 9
    assert all(sb.coverage == 1)
    assert len(sb.coverage) == 9

    # Successful Initialization with coverage
    sb = ShiftBlueprint(2, 9, "Morning", jnp.ones(9) * 0.9)
    assert sb.duration == 9

    # Unsuccessful Initialization (Wrong size coverage...)
    with pytest.raises(ValueError):
        sb = ShiftBlueprint(2, 9, "Morning", jnp.ones(10) * 0.9)


def test_get_shift_coverage_basic():
    sb = ShiftBlueprint(start_int=2, duration=9, name="Test Shift")
    day_idx = 1
    daily_intervals = 24
    max_days = 4

    coverage = sb.get_shift_coverage(day_idx, daily_intervals, max_days)

    expected_coverage = (
        jnp.zeros(daily_intervals * max_days).at[24 + 2 : 24 + 2 + 9].set(1)
    )

    assert jnp.array_equal(coverage, expected_coverage)


def test_params_inheritance_works_as_intended():
    p = Params("none")
    gd = GradientDescentParams()

    assert repr(p) == "Params(type='none')"
    assert repr(gd) == "Params(type='gradient_descent')"
    assert gd.is_type("gradient_descent")


def test_requirements_setup(local_tests_objects):
    shifts = local_tests_objects.get_object("shifts")
    days = 1
    daily_intervals = 24

    sr = ShiftRequirements(shifts=shifts, days=days, daily_intervals=daily_intervals)

    assert len(sr.shift_matrix) == daily_intervals * days
    assert sr.shift_matrix_cols == ("Morning-0", "Afternoon-0", "Night-0")

    # valid type set params
    sr.set_params(GradientDescentParams(learning_rate=0.2))
    assert sr.params["gradient_descent"].learning_rate == 0.2

    # invalid type set params
    with pytest.raises(ValueError) as e:
        sr.set_params(Params(type="invalid_type"))
    assert "invalid_type" in str(e.value)


def test_generate_has_correct_output(local_tests_objects):
    shifts = local_tests_objects.get_object("shifts")
    days = 7
    daily_intervals = 24
    interval_reqs = jnp.array(
        local_tests_objects.get_object("test_values")["hourly_reqs"]["cyclical_7x24"]
    )[0 : days * daily_intervals]

    sr = ShiftRequirements(shifts=shifts, days=days, daily_intervals=daily_intervals)
    sr.set_params(GradientDescentParams(learning_rate=0.1))
    sr.set_params(StochasticRoudningParams(section_size=5))
    w, gd_ev, sr_ev = sr.generate(interval_reqs=interval_reqs, seed=0)

    gd_params: GradientDescentParams = sr.params["gradient_descent"]
    sr_params: StochasticRoudningParams = sr.params["stochastic_rounding"]

    assert len(w) == sr.num_shifts
    assert len(gd_ev) == gd_params.num_iterations / gd_params.snapshot_length
    assert all(
        [
            set(ev_step.keys()) == set(["iteration_id", "error", "values"])
            for ev_step in gd_ev
        ]
    )
    assert len(sr_ev) > 0
    assert len(sr_ev) == jnp.ceil(sr.num_shifts / sr_params.section_size)
    assert all(
        [
            set(ev_step.keys()) == set(["iteration_id", "error", "values"])
            for ev_step in sr_ev
        ]
    )

import jax.numpy as jnp
import time
from typing import Dict, Tuple, Optional
from src.entities.data_classes import Shift

from src.algorythms.params import (
    GradientDescentParams,
    StochasticRoudningParams,
    Params,
)
from src.algorythms.optimization import gradient_descent, stochastic_rounding


class ShiftBlueprint:
    def __init__(
        self,
        shift: Shift,
        daily_intervals: int = 24,
        coverage: Optional[jnp.ndarray] = None,
    ):
        self.id = shift.id
        self.start_int = jnp.trunc(shift.start_time * daily_intervals / 24)
        self.duration = shift.duration
        self.coverage = coverage

        if coverage is None:
            self.coverage = jnp.ones(shift.duration)
        elif len(coverage) != shift.duration:
            raise ValueError("Coverage must be of length duration")
        elif jnp.any((coverage > 1.0) & (coverage < 0.0)):
            raise ValueError("Coverage must have values between 0 and 1")
        else:
            self.coverage = coverage

    def get_shift_coverage(
        self, day_idx: int, daily_intervals: int, max_days: int
    ) -> jnp.ndarray:
        start_index = day_idx * daily_intervals + int(self.start_int)
        end_index = min(start_index + int(self.duration), daily_intervals * max_days)
        coverage = self.coverage[0 : end_index - start_index].copy()
        return (
            jnp.zeros(daily_intervals * max_days)
            .at[start_index:end_index]
            .set(coverage)
        )


class ShiftRequirements:
    def __init__(
        self,
        shifts: list[ShiftBlueprint],
        days: int,
        daily_intervals: int,
        asymmetric_weight: float = 0.95,
    ) -> None:
        self.shifts = shifts
        self.days = days
        self.daily_intervals = daily_intervals
        self.asymmetric_weight = asymmetric_weight
        self.shift_matrix, self.shift_matrix_cols = self._generate_shift_matrix()
        self.params = {
            "gradient_descent": GradientDescentParams(
                learning_rate=0.1, num_iterations=10000, snapshot_length=100
            ),
            "stochastic_rounding": StochasticRoudningParams(
                section_size=len(self.shifts)
            ),
        }
        self.last_generated = dict()
        if daily_intervals < 1:
            raise ValueError("Number of daily intervals must be a greater than 1")
        if days < 1:
            raise ValueError("Number of days must be a greater than 1")
        if len(shifts) < 1:
            raise ValueError("Please add at least one Shift Blueprint")
        if not 0 <= asymmetric_weight <= 1:
            raise ValueError("Assymetric weight must be float between 0 and 1")

    def asymmetric_error_function(
        self, weights: jnp.ndarray, target: jnp.ndarray
    ) -> jnp.ndarray:
        asymmetric_weight = self.asymmetric_weight
        values = self.total_coverage(weights, self.shift_matrix)
        symmetric_error = jnp.mean((values - target) ** 2)
        asymmetric_error = jnp.mean(
            jnp.where(values - target < 0, (values - target) ** 2, 0)
        )
        return (
            1 - asymmetric_weight
        ) * symmetric_error + asymmetric_weight * asymmetric_error

    def set_params(self, new_params: Params) -> None:
        type = new_params.type
        if type in self.params.keys():
            self.params[type] = new_params
        else:
            raise ValueError("Invalid params type: %s", type)

    def generate_lower_bounds(self, lower_bounds: Dict[str, list[int]]) -> jnp.ndarray:
        split_columns = (name.split("-") for name in self.shift_matrix_cols)
        output = []
        for name, day in split_columns:
            pattern = lower_bounds.get(name, None)
            if pattern != None:
                output.append(pattern[int(day) % len(pattern)])
            else:
                output.append(0)
        return jnp.array(output)

    def generate(
        self,
        interval_reqs: jnp.ndarray,
        seed: int = int(time.time()),
        lower_bounds: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, list[dict], list[dict]]:

        # Initialize Weights and Evolution
        weights = jnp.zeros(
            self.num_shifts,
        )
        if lower_bounds != None:
            if len(lower_bounds) != self.num_shifts:
                raise ValueError(
                    "Lower bounds must have same length as number of shifts."
                )
            else:
                weights += lower_bounds

        # Gradient Descent
        print(f"--- GRADIENT DESCENT ---")
        weights, gd_evolution = gradient_descent(
            weights,
            interval_reqs,
            self.params["gradient_descent"],
            self.asymmetric_error_function,
            lower_bounded=True,
        )

        # Stochastic Rounding
        print(f"--- STOCHASTIC ROUNDING ---")
        weights, sr_evolution = stochastic_rounding(
            weights,
            interval_reqs,
            self.params["stochastic_rounding"],
            self.asymmetric_error_function,
            seed,
        )

        self.last_generated = {
            "weights": weights.copy(),
            "gd_evolution": gd_evolution.copy(),
            "sr_evolution": sr_evolution.copy(),
            "shift_matrix": self.shift_matrix.copy(),
            "interval_reqs": interval_reqs.copy(),
            "params": self.params.copy(),
            "random_seed": seed,
        }

        return weights, gd_evolution, sr_evolution

    # Initialization Method
    def _generate_shift_matrix(
        self,
    ) -> Tuple[jnp.ndarray, tuple[str]]:
        ids = [shift.id for shift in self.shifts]
        coverage_list = []
        shift_matrix_columns = []
        if len(ids) != len(set(ids)):
            raise ValueError("Shifts must have unique ids.")
        for day in range(self.days):
            for shift in self.shifts:
                if shift.start_int > self.daily_intervals:
                    raise ValueError(
                        "Shift start time must be less than daily intervals"
                    )
                coverage_list.append(
                    shift.get_shift_coverage(
                        day_idx=day,
                        daily_intervals=self.daily_intervals,
                        max_days=self.days,
                    ).reshape(-1, 1)
                )
                shift_matrix_columns.append(f"{shift.id}-{day}")

        return (jnp.concatenate(coverage_list, axis=1), tuple(shift_matrix_columns))

    # Static Methods
    @staticmethod
    def total_coverage(weights: jnp.ndarray, shift_matrix: jnp.ndarray):
        return jnp.array(shift_matrix).dot(weights)

    # Last Generated Properties
    def get_last_generated_item(self, item: str) -> jnp.ndarray:
        return self.last_generated[item]

    @property
    def num_shifts(self):
        return len(self.shifts) * self.days

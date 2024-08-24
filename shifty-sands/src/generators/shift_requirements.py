from typing import Any, Optional, Dict, List, Tuple, Callable

import time
import jax.numpy as jnp


from src.generators.generator import Generator
from src.entities.shifts import RequirementShift

from src.algorithms.params import (
    GradientDescentParams,
    StochasticRoundingParams,
    Params,
)
from src.algorithms.optimization import gradient_descent, stochastic_rounding
from src.algorithms.error_functions import mse_with_asym_neg_penalty


class SRGeneratorParams(Params):
    def __init__(
        self,
        daily_intervals=int,
        asymetric_weight=float,
        gradient_descent=GradientDescentParams,
        stochastic_rounding=StochasticRoundingParams,
    ):
        super().__init__(type="sr_generator")
        self.daily_intervals = daily_intervals
        self.asymetric_weight = asymetric_weight
        self.gradient_descent = gradient_descent
        self.stochastic_rounding = stochastic_rounding


class ShiftRequirements:
    def __init__(
        self,
        weights: jnp.ndarray,
        gd_evolution: jnp.ndarray,
        sr_evolution: jnp.ndarray,
        shift_matrix: jnp.ndarray,
        shift_matrix_columns: List[Dict],
        params: Dict[str, Params],
        interval_reqs: jnp.ndarray,
        days: int,
        random_seed: int,
    ):
        self.weights = weights
        self.gd_evolution = gd_evolution
        self.sr_evolution = sr_evolution
        self.shift_matrix = shift_matrix
        self.shift_matrix_columns = shift_matrix_columns
        self.params = params
        self.interval_reqs = interval_reqs
        self.days = days
        self.random_seed = random_seed

    @property
    def coverage(self) -> jnp.ndarray:
        return jnp.array(self.shift_matrix).dot(self.weights)

    @property
    def column_names(self) -> List[str]:
        return [col["name"] for col in self.shift_matrix_columns]

    @property
    def column_days(self) -> List[int]:
        return [col["day"] for col in self.shift_matrix_columns]

    @property
    def column_shifts(self) -> List[RequirementShift]:
        return [col["shift"] for col in self.shift_matrix_columns]


class ShiftRequirementsGenerator(Generator):
    def __init__(
        self,
        shifts: list[RequirementShift],
        days: int = 7,
        daily_intervals: int = 24,
        asymmetric_weight: float = 0.95,
    ) -> None:
        self.shifts = shifts
        self.days = days
        self.params = SRGeneratorParams(
            daily_intervals=daily_intervals,
            asymetric_weight=asymmetric_weight,
            gradient_descent=GradientDescentParams(
                learning_rate=0.1, num_iterations=10000, snapshot_length=100
            ),
            stochastic_rounding=StochasticRoundingParams(
                section_size=len(shifts), passes=25
            ),
        )

        if daily_intervals < 1:
            raise ValueError("Number of daily intervals must be a greater than 1")
        if days < 1:
            raise ValueError("Number of days must be a greater than 1")
        if len(shifts) < 1:
            raise ValueError("Please add at least one Shift")
        if not 0 <= asymmetric_weight <= 1:
            raise ValueError("Assymetric weight must be float between 0 and 1")

    def generate(
        self,
        interval_reqs: jnp.ndarray,
        seed: int = int(time.time()),
    ) -> ShiftRequirements:
        shift_matrix, shift_matrix_columns = self._generate_shift_matrix()
        value_func = self._generate_value_function(shift_matrix)
        loss_func = self._generate_loss_function(value_func)

        # Initialize Weights
        weights = jnp.zeros(
            self.num_shifts,
        )

        weights += self._generate_lower_bounds()

        # Gradient Descent
        print(f"--- GRADIENT DESCENT ---")
        weights, gd_evolution = gradient_descent(
            weights,
            interval_reqs,
            self.params.gradient_descent,
            loss_func,
            lower_bounded=True,
        )

        # Stochastic Rounding
        print(f"--- STOCHASTIC ROUNDING ---")
        weights, sr_evolution = stochastic_rounding(
            weights,
            interval_reqs,
            self.params.stochastic_rounding,
            loss_func,
            seed,
        )

        return ShiftRequirements(
            weights,
            gd_evolution,
            sr_evolution,
            shift_matrix,
            shift_matrix_columns,
            self.params,
            interval_reqs,
            self.days,
            seed,
        )

    def _generate_lower_bounds(self) -> jnp.ndarray:
        output = []
        for day in range(self.days):
            for shift in self.shifts:
                pattern = shift.lower_bounds
                if pattern != None:
                    output.append(pattern[int(day) % len(pattern)])
                else:
                    output.append(0)

        return jnp.array(output)

    def _generate_shift_matrix(self) -> Tuple[jnp.ndarray, tuple[Dict[str, Any]]]:
        ids = [shift.id for shift in self.shifts]
        coverage_list = []
        shift_matrix_columns = []
        if len(ids) != len(set(ids)):
            raise ValueError("Shifts must have unique ids.")
        for day in range(self.days):
            for shift in self.shifts:
                if shift.start_int > self.params.daily_intervals:
                    raise ValueError(
                        "Shift start time must be less than daily intervals"
                    )
                coverage_list.append(
                    shift.get_shift_coverage(
                        day_idx=day,
                        daily_intervals=self.params.daily_intervals,
                        max_days=self.days,
                    ).reshape(-1, 1)
                )
                shift_matrix_columns.append(
                    {"name": f"{shift.id}-{day}", "shift": shift, "day": day}
                )
        return (jnp.concatenate(coverage_list, axis=1), tuple(shift_matrix_columns))

    def _generate_value_function(self, shift_matrix: jnp.ndarray) -> Callable:
        def value_func(weights: jnp.ndarray) -> jnp.ndarray:
            return jnp.array(shift_matrix).dot(weights)

        return value_func

    def _generate_loss_function(self, value_func: Callable) -> Callable:
        def loss_func(weights: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
            return mse_with_asym_neg_penalty(
                value_func(weights), target, self.params.asymetric_weight
            )

        return loss_func

    def set_params(self, new_params: SRGeneratorParams) -> None:
        self.params = new_params

    @property
    def num_shifts(self):
        return len(self.shifts) * self.days

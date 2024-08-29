import jax.numpy as jnp
from functools import wraps


def error_function(func):
    @wraps(func)
    def wrapper(values, target, *args, **kwargs):
        # Check if inputs have the same shape
        if values.shape != target.shape:
            raise ValueError("Both predictions and targets must have the same shape.")

        # Call the original function
        return func(values, target, *args, **kwargs)

    return wrapper


@error_function
def mse(values: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((target - values) ** 2)


@error_function
def mse_with_asym_neg_penalty(
    values: jnp.ndarray, target: jnp.ndarray, asymmetric_weight: float
) -> jnp.ndarray:
    symmetric_error = jnp.mean((values - target) ** 2)
    asymmetric_error = jnp.mean(
        jnp.where(values - target < 0, (values - target) ** 2, 0)
    )
    return (
        1 - asymmetric_weight
    ) * symmetric_error + asymmetric_weight * asymmetric_error


@error_function
def mre_with_asym_neg_penalty(
    values: jnp.ndarray,
    target: jnp.ndarray,
    asymmetric_weight: float,
    safety_factor: int,
) -> jnp.ndarray:
    residuals = (values - target) / (target + (1 / safety_factor))
    symmetric_error = jnp.mean((residuals) ** 2)
    asymmetric_error = jnp.mean(jnp.where(residuals < 0, (values - target) ** 2, 0))
    return (
        1 - asymmetric_weight
    ) * symmetric_error + asymmetric_weight * asymmetric_error

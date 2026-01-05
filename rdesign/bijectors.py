import jax

from jax import Array, numpy as jnp
from distrax import Bijector


class Tanh(Bijector):
    """Tanh bijector mapping (-∞, ∞) to (-1, 1).

    Useful for constraining policy outputs to bounded intervals.
    """

    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Array) -> Array:
        x = jnp.clip(y, -0.995, 0.995)
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)

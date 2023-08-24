import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from ..utils import GradientOptimizer, OptState, OptParams
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: float
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.04
    sigma_decay: float = 0.999
    sigma_limit: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    mahi_lrate: float = 0.01


class MahiES(Strategy):
    def __init__(self, num_dims: int, popsize: int, opt_name: str = "adam"):
        """OpenAI-ES (Salimans et al. (2017)
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(num_dims, popsize)
        assert not self.popsize & 1, "Population size must be even"
        self.strategy_name = "MahiES"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(
            self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init,
            best_member=initialization,
        )
        return state

    def ask_strategy(
            self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        # z_plus = jax.random.normal(rng,(int(self.popsize // 2), self.num_dims))
        # z = jnp.concatenate([z_plus, -1.0 * z_plus])
        # z = jnp.concatenate([z_plus, -1.0 * z_plus, jnp.zeros((1, self.num_dims))])

        # z_plus = jax.random.normal(rng,(self.popsize - 1, self.num_dims))

        z_plus = jax.random.uniform(rng,(self.popsize - 1, self.num_dims), minval=-1.41421, maxval=1.41421)
        z = jnp.concatenate([z_plus, jnp.zeros((1, self.num_dims))])
        x = state.mean + state.sigma * z

        # z = jax.random.orthogonal(rng, self.num_dims)[:self.popsize - 1]
        # z = jnp.concatenate([z, jnp.zeros((1, self.num_dims))])
        # x = state.mean + state.sigma * z

        return x, state

    def tell_strategy(
            self,
            x: chex.Array,
            fitness: chex.Array,
            state: EvoState,
            params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state.mean) / state.sigma

        # Treat fitness as weights and compute dot product with noise
        weighted_fitness_dot = jnp.dot(noise.T, fitness[:-1])

        # Update the mean using the dot product result and bias term
        mean = weighted_fitness_dot + fitness[-1]

        sigma = state.sigma * params.sigma_decay
        # sigma = jnp.maximum(sigma, params.sigma_limit)

        return state.replace(mean=state.mean+mean, sigma=sigma)
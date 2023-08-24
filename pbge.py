import jax
import jax.numpy as jnp  # JAX NumPy
import numpy as np  # Ordinary NumPy
import wandb
from backprop import sl
from utils import helpers, models, evo
import chex
from args import get_args
from evosax import NetworkMapper, ParameterReshaper, FitnessShaper
from flax.core import FrozenDict

import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# cosine distance
def cosine(x, y):
    return jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x ** 2)) * jnp.sqrt(jnp.sum(x ** 2)))


def cosine2(x, y):
    return jnp.sum(x * y) / (jnp.sqrt(jnp.sum(x ** 2)) * jnp.sqrt(jnp.sum(y ** 2)))

# l2 distance
def l2(x, y):
    return -1 * jnp.sqrt(jnp.sum((x - y) ** 2))  # / jnp.sqrt(jnp.sum(x ** 2))


def l1(x, y):
    return -1 * jnp.sum(jnp.abs(x - y))


class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} s{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size,
                    args.seed, wandb.run.id)

        wandb.run.save()
        self.args = args

    def run(self, rng: chex.PRNGKey):
        train_ds, test_ds = sl.get_datasets(wandb.config.dataset.lower())
        rng, init_rng = jax.random.split(rng)

        learning_rate = wandb.config.lr
        momentum = wandb.config.momentum
        network = NetworkMapper[wandb.config.network_name](**wandb.config.network_config)

        state = sl.create_train_state(init_rng, network, learning_rate, momentum)
        param_reshaper = ParameterReshaper(state.params, n_devices=1)
        test_param_reshaper = ParameterReshaper(state.params, n_devices=1)
        strategy, es_params = evo.get_strategy_and_params(self.args.pop_size, param_reshaper.total_params, self.args)
        fit_shaper = FitnessShaper(centered_rank=True, z_score=True, w_decay=self.args.w_decay, maximize=True)
        server = strategy.initialize(init_rng, es_params)
        server = server.replace(mean=test_param_reshaper.network_to_flat(state.params))
        del init_rng  # Must not be used anymore.

        num_epochs = wandb.config.n_rounds
        batch_size = wandb.config.batch_size
        X, y = jnp.array(train_ds['image']), jnp.array(train_ds['label'])

        for epoch in range(1, num_epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng, rng_ask = jax.random.split(rng, 3)
            # Run an optimization step over a training batch
            target_state, loss, acc = sl.train_epoch(state, X, y, batch_size, input_rng)
            # Evaluate on the test set after each training epoch
            target_server = param_reshaper.network_to_flat(target_state.params)
            search_range = jnp.linalg.norm(server.mean - target_server)

            x, server = strategy.ask(rng_ask, server.replace(sigma=self.args.sigma_init), es_params)
            # x, server = strategy.ask(rng_ask, server.replace(sigma=search_range * 4.0), es_params)

            # grad_signal = target_server - server.mean
            # grad_signal_mag = jnp.linalg.norm(grad_signal)
            # grad_norm = grad_signal / grad_signal_mag
            #
            # x_signal = x - server.mean
            # x_mag = jnp.linalg.norm(x_signal, axis=1)
            # x_norm = x_signal / x_mag[:, None]
            # fitness = jax.vmap(cosine2, in_axes=(0, None))(x_norm, grad_norm)
            fitness = jax.vmap(l2, in_axes=(0, None))(x - server.mean, target_server - server.mean)
            fitness = fit_shaper.apply(x, fitness)
            bd = jnp.mean((server.mean - target_server)**2)
            server = strategy.tell(x, fitness, server, es_params)
            ad = jnp.mean((server.mean - target_server)**2)

            # server = strategy.tell(x, fitness, server, es_params)
            # state = state.replace(params=FrozenDict(test_param_reshaper.reshape_single_net(server.mean)))
            state = sl.update_train_state(learning_rate, momentum, test_param_reshaper.reshape_single_net(server.mean))
            test_loss, test_accuracy = sl.eval_model(state.params, test_ds, input_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Train Loss': loss,
                'Test Accuracy': test_accuracy,
                'Train Accuracy': acc,
                'Global Accuracy': test_accuracy,
                'Distance': bd,
                'Compression Error': ad,
                'Compression Rate': 1 - (ad / bd),
            })


def run():
    print(jax.devices())
    args = get_args()
    config = helpers.load_config(args.config)
    wandb.init(project='evofed-new', config=args)
    wandb.config.update(config)
    args = wandb.config
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init, rng_run = jax.random.split(rng, 3)
    manager = TaskManager(rng_init, args)
    manager.run(rng_run)


SWEEPS = {
    'cifar-bp': 'bc4zva3u',
    'cifar-bp2': '82la1zw0',
    'fmnits-mah': '1yksrmvs',
    'cifar-mah': 'mtheusi1',
}

if __name__ == '__main__':
    run()
    # wandb.agent(SWEEPS['cifar-mah'], function=run, project='evofed', count=10)

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


from jax import lax


def vector_projection(a, b):
    b_normalized = b / jnp.linalg.norm(b)
    projection = jnp.dot(a, b_normalized) * b_normalized
    return projection

def cosine(x, y):
    dot_product = lax.dot(x, y)
    norm_x = lax.sqrt(lax.dot(x, x))
    norm_y = lax.sqrt(lax.dot(y, y))
    return -1 * dot_product / (norm_x * norm_y)


def normalize_vector(x):
    norm = jnp.linalg.norm(x)
    return x / norm

# l2 distance
def l2(x, y):
    return -1 * jnp.sqrt(jnp.sum((x - y) ** 2))


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
        param_reshaper = ParameterReshaper(state.params, n_devices=self.args.n_devices)
        test_param_reshaper = ParameterReshaper(state.params, n_devices=1)
        parts = wandb.config.parts
        padding = parts - param_reshaper.total_params % parts
        strategy, es_params = evo.get_strategy_and_params(self.args.pop_size, (param_reshaper.total_params + padding) // parts, self.args)
        fit_shaper = FitnessShaper(centered_rank=True, z_score=True, w_decay=self.args.w_decay, maximize=True)
        server = strategy.initialize(init_rng, es_params)
        flat_param = test_param_reshaper.network_to_flat(state.params)

        # add zero padding to flat_param to be dividable by parts
        flat_param = jnp.concatenate([flat_param, jnp.zeros(padding)])
        print(flat_param.shape)

        server = [server.replace(mean=flat_param.reshape(parts, -1)[i]) for i in range(parts)]
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
            target_server = jnp.concatenate([target_server, jnp.zeros(padding)])
            # search_range = jnp.linalg.norm(server.mean - target_server)

            x_server = [strategy.ask(rng_ask, server[i].replace(sigma=self.args.sigma_init), es_params) for i in range(parts)]
            x, server = jnp.array([x[0] for x in x_server]), [x[1] for x in x_server]
            means = jnp.array([s.mean for s in server])

            # split x and target_server into parts
            target_server = target_server.reshape(parts, -1)

            bd = jnp.sum(server[0].mean - target_server[0])
            # fitness = jax.vmap(jax.vmap(l2, in_axes=(0, None)))(jax.vmap(lambda x,y: x - y)(x, means), target_server - means)
            fitness = jax.vmap(jax.vmap(l2, in_axes=(0, None)))(x, target_server)
            fitness = jax.vmap(fit_shaper.apply)(x, fitness)
            server = [strategy.tell(x[i], fitness[i], server[i], es_params) for i in range(parts)]
            ad = jnp.sum(server[0].mean - target_server[0])
            mean = jnp.concatenate([server.mean for server in server])[:-padding]
            state = sl.update_train_state(learning_rate, momentum, test_param_reshaper.reshape_single_net(mean))
            test_loss, test_accuracy = sl.eval_model(state.params, test_ds, input_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Train Loss': loss,
                'Test Accuracy': test_accuracy,
                'Train Accuracy': acc,
                'Global Accuracy': test_accuracy,
                'Before': bd,
                'After': ad,
                'Update': (ad - bd),
            })


def run():
    print(jax.devices())
    args = get_args()
    config = helpers.load_config(args.config)
    wandb.init(project='evofed-publish', config=args, save_code=True, notes=os.path.basename(__file__))
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
    'cifar-mah-part': 'u4of6nir',
    'cifar-fedpart': 'd4nm9bgr',
    'cifar-part3': 'llxwakb7',
}

if __name__ == '__main__':
    run()
    # wandb.agent(SWEEPS['cifar-part3'], function=run, project='evofed', count=20)

import chex
import jax
import jax.numpy as jnp  # JAX NumPy
import tensorflow_datasets as tfds  # TFDS for MNIST
import wandb
from evosax import NetworkMapper
from backprop import sl
from args import get_args
from utils import helpers, evo
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
    return -1 * jnp.sqrt(jnp.sum((x - y) ** 2))

def l1(x, y):
    return -1 * jnp.sum(jnp.abs(x - y))

def pnorm(x, y, p):
    x = jnp.abs(x - y)
    return -1 * jnp.sum(x ** p) ** (1 / p)


def max_dist(x, y):
    return -1 * 0.02 * jnp.max(jnp.abs(x - y)) + 0.98 * l2(x, y)

# def l2_std(x, y):
#     return l2(x, y) +
def sparsify(array, percentage):
    original = array
    array = jnp.abs(array.flatten())
    array = jnp.sort(array)
    threshold = array[int(len(array) * percentage)]
    array = jnp.where(jnp.abs(original) < threshold, 0, original)
    return array

def quantize(array, min_val, max_val, n_bits):
    # max_val = array.max()
    # min_val = array.min()
    step = (max_val - min_val) / (2 ** n_bits - 1)
    array = ((array - min_val) / step).round()
    return array


# dequantization array
def dequantize(array, min_val, max_val, n_bits):
    step = (max_val - min_val) / (2 ** n_bits - 1)
    array = array * step + min_val
    return array

class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} c{} s{} p{} r{} q{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size, args.n_clients,
                    args.seed,
                    args.percentage,
                    args.rank_factor,
                    args.quantize_bits,
                    wandb.run.id)
        wandb.run.save()
        # self.train_ds, self.test_ds = sl.get_datasets_non_iid(args.dataset, args.n_clients) \
        #     if args.dist == 'NON-IID' else sl.get_datasets_iid(args.dataset, args.n_clients)
        self.train_ds, self.test_ds = sl.get_fed_datasets(args.dataset, args.n_clients, 20, args.dist == 'IID')
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        self.learning_rate = wandb.config.lr
        self.momentum = wandb.config.momentum
        network = NetworkMapper[wandb.config.network_name](**wandb.config.network_config)

        self.state = sl.create_train_state(init_rng, network, self.learning_rate, self.momentum)
        self.param_reshaper = ParameterReshaper(self.state.params, n_devices=1)
        self.test_param_reshaper = ParameterReshaper(self.state.params, n_devices=1)
        self.strategy, self.es_params = evo.get_strategy_and_params_cma(args.pop_size, self.param_reshaper.total_params,
                                                                    args)
        self.fit_shaper = FitnessShaper(centered_rank=args.centered_rank, z_score=args.z_score,
                                        w_decay=args.w_decay, maximize=args.maximize, rank_factor=args.rank_factor)
        server = self.strategy.initialize(init_rng, self.es_params)
        self.server = server.replace(mean=self.test_param_reshaper.network_to_flat(self.state.params))
        del init_rng  # Must not be used anymore.

        self.param_count = sum(x.size for x in jax.tree_leaves(self.state.params))
        self.num_epochs = wandb.config.n_rounds
        self.batch_size = wandb.config.batch_size
        self.n_clients = args.n_clients

        min_cut = 10000
        self.X = jnp.array([train['image'][:min_cut] for train in self.train_ds])
        self.y = jnp.array([train['label'][:min_cut] for train in self.train_ds])
        self.args = args
        self.n_bits = args.quantize_bits

    def run(self, rng: chex.PRNGKey):
        for epoch in range(0, self.num_epochs + 1):

            rng, input_rng, rng_ask = jax.random.split(rng, 3)
            clients, _, _ = jax.vmap(sl.train_epoch, in_axes=(None, 0, 0, None, None))(self.state,
                                                                                            self.X,
                                                                                            self.y,
                                                                                            self.batch_size, input_rng)

            target_server = jax.vmap(self.param_reshaper.network_to_flat)(clients.params)
            x, self.server = self.strategy.ask(rng_ask, self.server, self.es_params)
            fitness = jax.vmap(jax.vmap(l2, in_axes=(0, None)), in_axes=(None, 0))(x-self.server.mean, target_server-self.server.mean)
            fitness = jax.vmap(self.fit_shaper.apply, in_axes=(None, 0))(x, fitness)
            # fitness = jax.vmap(sparsify, in_axes=(0, None))(fitness, self.args.percentage)
            # fitness = jax.vmap(quantize, in_axes=(0, None, None, None))(fitness, -0.5, 0.5, self.n_bits)
            fitness = fitness.mean(0)
            # fitness = dequantize(fitness, -0.5, 0.5, self.n_bits)
            # fitness = sparsify(fitness, self.args.percentage)
            origin = self.server.mean.copy()
            self.server = self.strategy.tell(x, fitness, self.server, self.es_params)
            self.state = self.state.replace(params=FrozenDict(self.test_param_reshaper.reshape_single_net(self.server.mean)))
            # sl.update_learning_rate(self.state, epoch)

            rng, eval_rng = jax.random.split(rng)
            test_loss, test_accuracy = sl.eval_model(self.state.params, self.test_ds, eval_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Global Accuracy': test_accuracy,
                'Distance': jnp.mean((origin - target_server) ** 2),
                'Compression Error': jnp.mean((self.server.mean - target_server)**2),
                # 'Communication': epoch * 2 * self.args.pop_size * (1 - self.args.percentage) * (1 + np.log2(self.args.pop_size)),
                # 'Communication': epoch * 4 * self.args.pop_size * (1 - self.args.percentage) * (np.log2(self.args.pop_size * np.sqrt((1 - self.args.percentage) * 1/self.args.rank_factor))),
                'Communication': epoch * 2 * self.args.pop_size * (1 - self.args.percentage) * ((self.n_bits + jnp.log2(self.args.pop_size))/ 32),
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


if __name__ == '__main__':
    run()
    # wandb.agent('tjjj64sq', function=run, project='evofed', count=20)

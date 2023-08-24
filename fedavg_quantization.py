import chex
import jax
import jax.numpy as jnp  # JAX NumPy
import numpy as np
import tensorflow_datasets as tfds  # TFDS for MNIST
import wandb
from evosax import NetworkMapper
from backprop import sl
from args import get_args
from utils import helpers
from flax.core import FrozenDict
from evosax import ParameterReshaper
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# compress the array with quantization based on array distribution
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

def sparsify(array, percentage):
    original = array
    array = jnp.abs(array.flatten())
    array = jnp.sort(array)
    threshold = array[int(len(array) * percentage)]
    array = jnp.where(jnp.abs(original) < threshold, 0, original)
    return array

# L2 distance
def l2(x, y):
    return -1 * jnp.sqrt(jnp.sum((x - y) ** 2))  # / jnp.sqrt(jnp.sum(x ** 2))


class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} c{} s{} q{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size, args.n_clients,
                    args.seed, args.quantize_bits, wandb.run.id)
        wandb.run.save()
        self.train_ds, self.test_ds = sl.get_fed_datasets(args.dataset, args.n_clients, 2, args.dist == 'IID')

        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        self.learning_rate = wandb.config.lr
        self.momentum = wandb.config.momentum
        network = NetworkMapper[wandb.config.network_name](**wandb.config.network_config)

        self.state = sl.create_train_state(init_rng, network, self.learning_rate, self.momentum)
        del init_rng  # Must not be used anymore.

        self.param_count = sum(x.size for x in jax.tree_leaves(self.state.params))
        self.num_epochs = wandb.config.n_rounds
        self.batch_size = wandb.config.batch_size
        self.client_epoch = wandb.config.client_epoch
        self.n_clients = args.n_clients
        min_cut = 10000
        # if args.dataset == 'mnist':
        #     min_cut = 5421

        self.X = jnp.array([train['image'][:min_cut] for train in self.train_ds])
        self.y = jnp.array([train['label'][:min_cut] for train in self.train_ds])
        self.args = args
        self.n_bits = args.quantize_bits
        self.param_reshaper = ParameterReshaper(self.state.params, n_devices=1)

    def run(self, rng: chex.PRNGKey):
        for epoch in range(0, self.num_epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            # clients = [self.state for i in range(5)]
            clients, loss, acc = jax.vmap(sl.train_epoch, in_axes=(None, 0, 0, None, None))(self.state,
                                                                                            self.X,
                                                                                            self.y,
                                                                                            self.batch_size, input_rng)
            for c_epoch in range(self.client_epoch):
                input_rng, c_rng = jax.random.split(input_rng)
                clients, loss, acc = jax.vmap(sl.train_epoch, in_axes=(0, 0, 0, None, None))(clients,
                                                                                             self.X,
                                                                                             self.y,
                                                                                             self.batch_size, c_rng)
                wandb.log({
                    'Epoch': epoch * self.client_epoch + c_epoch,
                    'Train Loss': loss.mean(),
                    'Train Accuracy': acc.mean(),
                })

            server = self.param_reshaper.network_to_flat(self.state.params)
            target_server = jax.vmap(self.param_reshaper.network_to_flat)(clients.params)
            target_server = (target_server - server)
            min_val, max_val = jax.vmap(jnp.min)(target_server), jax.vmap(jnp.max)(target_server)
            target_server = jax.vmap(sparsify, in_axes=(0, None))(target_server, self.args.percentage)

            # target_server = jax.vmap(quantize, in_axes=(0, 0, 0, None))(target_server, min_val, max_val, self.n_bits)
            # target_server = jax.vmap(dequantize, in_axes=(0, 0, 0, None))(target_server, min_val, max_val, self.n_bits)
            target_server = jax.vmap(jnp.mean)(target_server.T)
            # target_server = jax.vmap(quantize, in_axes=(0, None))(target_server, self.n_bits)
            # target_server = dequantize(target_server, min_val.mean(), max_val.mean(), self.n_bits)
            target_server = sparsify(target_server, self.args.percentage)

            target_server = target_server + server
            params = self.param_reshaper.reshape_single_net(target_server)
            self.state = self.state.replace(params=FrozenDict(params))
            rng, eval_rng = jax.random.split(rng)
            test_loss, test_accuracy = sl.eval_model(params, self.test_ds, eval_rng)
            remining_params = self.param_count * (1 - self.args.percentage)

            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Global Accuracy': test_accuracy,
                # 'Communication': epoch * 2 * self.param_count / (32 / self.n_bits),
                'Communication': epoch * 2 * remining_params * ((self.n_bits + np.log2(self.param_count))/ 32),
            })

def run():
    print(jax.devices())
    args = get_args()
    config = helpers.load_config(args.config)
    wandb.init(project='evofed-publish', config=args)
    wandb.config.update(config)
    args = wandb.config
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init, rng_run = jax.random.split(rng, 3)
    manager = TaskManager(rng_init, args)
    manager.run(rng_run)


if __name__ == '__main__':
    # wandb.agent('tdt4lz81', function=run, project='evofed', count=10)
    run()

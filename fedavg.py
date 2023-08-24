import chex
import jax
import jax.numpy as jnp  # JAX NumPy
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
def flatten_params(params):
    # Flatten each sub-tree of params into a single vector
    flat_params = jax.tree_map(lambda x: x.ravel(), params)

    # Concatenate all vectors together into a single vector
    concatenated_params = jnp.concatenate([param for param in jax.tree_leaves(flat_params)])

    return concatenated_params


def mean(x):
    return jnp.mean(x, axis=0)

class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} c{} s{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size, args.n_clients,
                    args.seed, wandb.run.id)
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
        del init_rng  # Must not be used anymore.

        self.param_count = sum(x.size for x in jax.tree_leaves(self.state.params))
        self.num_epochs = wandb.config.n_rounds
        self.batch_size = wandb.config.batch_size
        self.client_epoch = wandb.config.client_epoch
        self.param_reshaper = ParameterReshaper(self.state.params, n_devices=1)

        self.n_clients = args.n_clients
        min_cut = 10000
        # if args.dataset == 'mnist':
        #     min_cut = 5421 * 2
        if len(self.train_ds) == self.n_clients:
            self.X = jnp.array([train['image'][:min_cut] for train in self.train_ds])
            self.y = jnp.array([train['label'][:min_cut] for train in self.train_ds])
        else:
            self.X = jnp.array([self.train_ds['image'] for _ in range(self.n_clients)])
            self.y = jnp.array([self.train_ds['label'] for _ in range(self.n_clients)])
        self.args = args

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


            target_server = jax.vmap(self.param_reshaper.network_to_flat)(clients.params)
            target_server = jax.vmap(jnp.mean)(target_server.T)
            params = self.param_reshaper.reshape_single_net(target_server)
            self.state = self.state.replace(params=FrozenDict(params))
            rng, eval_rng = jax.random.split(rng)
            test_loss, test_accuracy = sl.eval_model(params, self.test_ds, eval_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Global Accuracy': test_accuracy,
                'Communication': epoch * 2 * self.param_count,
            })
            # print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            #     epoch, test_loss, test_accuracy * 100))


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
    run()
    # wandb.agent('tdt4lz81', function=run, project='evofed', count=10)
    # wandb.agent('42wg77vr', function=run, project='evofed', count=10)
    # wandb.agent('57p3byl2', function=run, project='evofed', count=10) #fedavg
    # wandb.agent('igjobuu4', function=run, project='evofed', count=10) #fedavg1

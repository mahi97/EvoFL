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
num_devices = jax.local_device_count()

class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} c{} s{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size, args.n_clients,
                    args.seed, wandb.run.id)
        wandb.run.save()
        self.train_ds, self.test_ds = sl.get_fed_datasets(args.dataset, args.n_clients, 20, args.dist == 'IID')
        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)

        self.learning_rate = wandb.config.lr
        self.momentum = wandb.config.momentum
        network = NetworkMapper[wandb.config.network_name](**wandb.config.network_config)
        self.state = sl.create_train_state(init_rng, network, self.learning_rate, self.momentum)
        self.param_reshaper = ParameterReshaper(self.state.params, n_devices=num_devices)
        self.test_param_reshaper = ParameterReshaper(self.state.params, n_devices=1)
        self.param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.state.params))
        self.parts = args.parts
        self.padding = self.parts - self.param_reshaper.total_params % self.parts
        self.strategy, self.es_params = evo.get_strategy_and_params(args.pop_size, (self.param_reshaper.total_params + self.padding) // self.parts, args)
        self.fit_shaper = FitnessShaper(centered_rank=args.centered_rank, z_score=args.z_score, w_decay=args.w_decay, maximize=args.maximize)
        server = self.strategy.initialize(init_rng, self.es_params)
        flat_param = self.test_param_reshaper.network_to_flat(self.state.params)
        flat_param = jnp.concatenate([flat_param, jnp.zeros(self.padding)])
        self.server = [server.replace(mean=flat_param.reshape(self.parts, -1)[i]) for i in range(self.parts)]
        # add zero padding to flat_param to n jax.tree_leaves(self.state.params))
        self.num_epochs = wandb.config.n_rounds
        self.batch_size = wandb.config.batch_size
        self.n_clients = args.n_clients

        min_cut = 10000
        self.X = jnp.array([train['image'][:min_cut] for train in self.train_ds])
        self.y = jnp.array([train['label'][:min_cut] for train in self.train_ds])
        self.args = args

        del init_rng  # Must not be used anymore.


    def run(self, rng: chex.PRNGKey):
        for epoch in range(0, self.num_epochs + 1):

            rng, input_rng, rng_ask = jax.random.split(rng, 3)
            clients, loss, acc = jax.vmap(sl.train_epoch, in_axes=(None, 0, 0, None, None))(self.state,
                                                                                            self.X,
                                                                                            self.y,
                                                                                            self.batch_size, input_rng)

            # for c_epoch in range(self.args.client_epoch):
            #     input_rng, c_rng = jax.random.split(input_rng)
            #     clients, loss, acc = jax.vmap(sl.train_epoch, in_axes=(0, 0, 0, None, None))(clients,
            #                                                                                  self.X,
            #                                                                                  self.y,
            #                                                                                  self.batch_size, c_rng)

            target_server = jax.vmap(self.param_reshaper.network_to_flat)(clients.params)
            target_server = jax.vmap(jnp.concatenate)([target_server, jnp.zeros([self.n_clients, self.padding])])
            x_server = [self.strategy.ask(rng_ask, self.server[i].replace(sigma=self.args.sigma_init), self.es_params) for i in range(self.parts)]
            x, self.server = jnp.array([x[0] for x in x_server]), [x[1] for x in x_server]
            # x, self.server = self.strategy.ask(rng_ask, self.server, self.es_params)
            # split x and target_server into 4 parts
            target_server = target_server.reshape(self.n_clients, self.parts, -1)

            fitness = jax.vmap(jax.vmap(jax.vmap(l2, in_axes=(0, None))), in_axes=(None, 0))(x, target_server)
            # fitness = self.fit_shaper.apply(x, -1.0 * jnp.linalg.norm(fitness, axis=0))
            # fitness = jax.vmap(jnp.meanjax.vmap(self.fit_shaper.apply)(x, fitness.mean(0))
            fitness = jax.vmap(jax.vmap(self.fit_shaper.apply), in_axes=(None, 0))(x, fitness).mean(axis=0)
            # self.server = self.strategy.tell(x, fitness, self.server, self.es_params)
            self.server = [self.strategy.tell(x[i], fitness[i], self.server[i], self.es_params) for i in range(self.parts)]

            # self.state = self.state.replace(params=FrozenDict(self.test_param_reshaper.reshape_single_net(self.server.mean)))

            mean = jnp.concatenate([server.mean for server in self.server])[:-self.padding]
            self.state = self.state.replace(params=FrozenDict(self.test_param_reshaper.reshape_single_net(mean)))
            rng, eval_rng = jax.random.split(rng)
            test_loss, test_accuracy = sl.eval_model(self.state.params, self.test_ds, eval_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Global Accuracy': test_accuracy,
                'Communication': epoch * 2 * self.args.pop_size * self.parts,
            })


def run():
    print(jax.devices())
    args = get_args()
    config = helpers.load_config(args.config)
    wandb.init(project='evofed-publish', config=args, save_code=True, notes=os.path.basename(__file__))
    wandb.config.update(config, allow_val_change=True)
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
    'cifar-fedpart2': 'mf4es3wq',
    'cifar-fedpart': 'd4nm9bgr',
    'cifar-fedpart4': 'przlfcf8',
    'cifar-fedpart3': 'xv6ne4jw',
    'cifar100-fedpart': 'mt9fse4u',
    'cifar100-fedpart2': 'nsx80v02',
    'cifar100-fedpart3': '5j57zzkf',
}

if __name__ == '__main__':
    run()
    # wandb.agent(SWEEPS['cifar100-fedpart3'], function=run, project='evofed', count=20)

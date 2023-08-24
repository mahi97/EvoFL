import jax
import jax.numpy as jnp  # JAX NumPy
import wandb
from backprop import sl
from utils import helpers, models
import chex
from args import get_args
from evosax import NetworkMapper
from jax.tree_util import tree_leaves
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} b{} s{} -- {}' \
            .format(args.dataset, args.algo,
                    args.dist,
                    args.batch_size,
                    args.seed, wandb.run.id)

        wandb.run.save()

    def run(self, rng: chex.PRNGKey):
        train_ds, test_ds = sl.get_datasets(wandb.config.dataset.lower())
        rng, init_rng = jax.random.split(rng)

        learning_rate = wandb.config.lr
        momentum = wandb.config.momentum
        network = NetworkMapper[wandb.config.network_name](**wandb.config.network_config)
        state = sl.create_train_state(init_rng, network, learning_rate, momentum)
        print(sum(x.size for x in jax.jax.tree_util.tree_leaves(state.params)))
        del init_rng  # Must not be used anymore.

        num_epochs = wandb.config.n_rounds
        batch_size = wandb.config.batch_size
        X, y = jnp.array(train_ds['image']), jnp.array(train_ds['label'])

        for epoch in range(1, num_epochs + 1):
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            state, loss, acc = sl.train_epoch(state, X, y, batch_size, input_rng)
            # Evaluate on the test set after each training epoch
            test_loss, test_accuracy = sl.eval_model(state.params, test_ds, input_rng)
            wandb.log({
                'Round': epoch,
                'Test Loss': test_loss,
                'Train Loss': loss,
                'Test Accuracy': test_accuracy,
                'Train Accuracy': acc,
                'Global Accuracy': test_accuracy,
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
    'cifar100-bp3': 'y8xesds8',
    'cifar100-bp4': 'uqepglhq',
    'cifar100-bp5': 'gdme5tyq',
}

if __name__ == '__main__':
    # wandb.agent(SWEEPS['cifar100-bp5'], function=run, project='evofed', count=10)
    run()
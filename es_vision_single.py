import jax
import chex
import jax.numpy as jnp

from evosax import ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.problems import VisionFitness, FederatedVisionFitness
import utils
from args import get_args
import wandb
from utils import models, evo, helpers


class TaskManager:
    def __init__(self, rng: chex.PRNGKey, args):
        wandb.run.name = '{}-{}-{} p{} b{} s{} -- {}'.format(args.dataset, args.strategy,
                                                                 args.dist, args.pop_size,
                                                                 args.batch_size,
                                                                 args.seed, wandb.run.id)
        wandb.run.save()
        self.args = args
        self.network = NetworkMapper[args.network_name](**args.network_config)
        params = self.network.init(rng, jnp.zeros(args.pholder), rng=rng)

        self.param_reshaper = ParameterReshaper(params, n_devices=1)
        self.test_param_reshaper = ParameterReshaper(params, n_devices=1)
        self.strategy, self.es_params = evo.get_strategy_and_params(args.pop_size,
                                                                    self.param_reshaper.total_params,
                                                                    args)

        # Set up the dataloader for batch evaluations (may take a sec)
        self.train_evaluator = VisionFitness(args.dataset, batch_size=args.batch_size, test=False,
                                                      n_devices=1)
        self.test_evaluator = VisionFitness(args.dataset, batch_size=10_000, test=True, n_devices=1)

        self.train_evaluator.set_apply_fn(self.param_reshaper.vmap_dict, self.network.apply)
        self.test_evaluator.set_apply_fn(self.test_param_reshaper.vmap_dict, self.network.apply)

        self.fit_shaper = FitnessShaper(centered_rank=True, z_score=True, w_decay=args.w_decay, maximize=True)

    def run(self, rng: chex.PRNGKey):
        rng, rng_client_init = jax.random.split(rng, 2)
        server = self.strategy.initialize(rng_client_init, self.es_params)

        for rnd in range(self.args.n_rounds):

            rng, rng_eval, rng_ask = jax.random.split(rng, 3)
            x, server = self.strategy.ask(rng_ask, server, self.es_params)
            reshaped_params = self.param_reshaper.reshape(x)
            rng, rng_evals = jax.random.split(rng_eval, 2)
            train_loss, train_acc = self.train_evaluator.rollout(rng_evals, reshaped_params)
            fit_re = self.fit_shaper.apply(x, train_loss.squeeze())
            server = self.strategy.tell(x, fit_re, server, self.es_params)
            # Evaluating the server performance
            rng, rng_server_ask, rng_server_eval = jax.random.split(rng, 3)
            # x, _ = self.strategy.ask(rng_server_ask, server, self.es_params)
            server_mean_params = server.mean.reshape(1, -1)
            server_reshaped_test_params = self.test_param_reshaper.reshape(server_mean_params)
            _, test_acc = self.test_evaluator.rollout(rng_server_eval, server_reshaped_test_params)
            wandb.log({
                'Round': rnd,
                'Global Accuracy': test_acc.squeeze()
            })


def run():
    print(jax.devices())
    args = get_args()
    config = helpers.load_config(args.config)
    wandb.init(project='evofed-seeds', config=args)
    wandb.config.update(config)
    args = wandb.config
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init, rng_run = jax.random.split(rng, 3)
    manager = TaskManager(rng_init, args)
    manager.run(rng_run)


if __name__ == '__main__':
    # wandb.agent('znfslak7', function=run, project='evofed', count=10)
    run()
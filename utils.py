import jax.numpy as jnp
from evosax import Strategies, NetworkMapper
import gymnax
from brax import envs


def get_network_and_pholder(task, args):
    if task in ['MNIST', 'FMNIST']:
        return NetworkMapper['CNN'](
            depth_1=1,
            depth_2=1,
            features_1=8,
            features_2=16,
            kernel_1=5,
            kernel_2=5,
            strides_1=1,
            strides_2=1,
            num_linear_layers=0,
            num_output_units=10,
        ), jnp.zeros((1, 28, 28, 1))
    elif task == 'CIFAR10':
        return NetworkMapper['CNN'](
            depth_1=1,
            depth_2=1,
            features_1=64,
            features_2=128,
            kernel_1=5,
            kernel_2=5,
            strides_1=1,
            strides_2=1,
            num_linear_layers=1,
            num_hidden_units=256,
            num_output_units=10,
        ), jnp.zeros((1, 32, 32, 3))
    elif task in ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Asterix-MinAtar']:
        env, env_param = gymnax.make(task)
        pholder = jnp.zeros(env.observation_space(env_param).shape)
        print(env.observation_space(env_param).shape, env.num_actions)
        if args.recurrent:
            network = NetworkMapper["LSTM"](
                num_hidden_units=32,
                num_output_units=env.num_actions,
                output_activation="categorical",
            )
        else:
            network = NetworkMapper["MLP"](
                num_hidden_units=64,
                num_hidden_layers=2,
                num_output_units=env.num_actions,
                hidden_activation="relu",
                output_activation="categorical",
            )
        return network, pholder
    elif task == ['Pendulum-v1', 'MountainCarContinuous-v0']:
        env, env_param = gymnax.make(task)
        pholder = jnp.zeros(env.observation_space(env_param).shape)
        if args.recurrent:
            network = NetworkMapper["LSTM"](
                num_hidden_units=32,
                num_output_units=env.num_actions,
                output_activation="gaussian",
            )
        else:
            network = NetworkMapper["MLP"](
                num_hidden_units=64,
                num_hidden_layers=2,
                num_output_units=env.num_actions,
                hidden_activation="relu",
                output_activation="gaussian",
            )
        return network, pholder
    elif task in [
            "ant",
            "halfcheetah",
            "hopper",
            "humanoid",
            "reacher",
            "walker2d",
            "fetch",
            "grasp",
            "ur5e",
        ]:
        env = envs.create(env_name=task)
        pholder = jnp.zeros((1, env.observation_size))

        if args.recurrent:
            network = NetworkMapper["LSTM"](
                num_hidden_units=32,
                num_output_units=env.action_size,
                output_activation="tanh",
            )
        else:
            network = NetworkMapper["MLP"](
                num_hidden_units=32,
                num_hidden_layers=4,
                num_output_units=env.action_size,
                hidden_activation="tanh",
                output_activation="tanh",
            )
        return network, pholder
    else:
        print('ERROR Task is not supported')


def get_strategy_and_params(pop_size, num_dims, args):
    MyStrategy = Strategies[args.strategy]
    strategy = MyStrategy(popsize=pop_size, num_dims=num_dims, opt_name=args.opt_name)
    es_params = strategy.default_params
    if args.strategy == 'OpenES':
        # Update basic parameters of PGPE strategy
        es_params = strategy.default_params.replace(
            sigma_init=args.sigma_init,  # Initial scale of isotropic Gaussian noise
            sigma_decay=args.sigma_decay,  # Multiplicative decay factor
            sigma_limit=args.sigma_limit,  # Smallest possible scale
            init_min=args.init_min,  # Range of parameter mean initialization - Min
            init_max=args.init_max,  # Range of parameter mean initialization - Max
            clip_min=args.clip_min,  # Range of parameter proposals - Min
            clip_max=args.clip_max  # Range of parameter proposals - Max
        )

        # Update optimizer-specific parameters of Adam
        es_params = es_params.replace(opt_params=es_params.opt_params.replace(
            lrate_init=args.lr_init,  # Initial learning rate
            lrate_decay=args.lrate_init,  # Multiplicative decay factor
            lrate_limit=args.lrate_limit,  # Smallest possible lrate
            beta_1=args.beta_1,  # Adam - beta_1
            beta_2=args.beta_2,  # Adam - beta_2
            eps=args.eps,  # eps constant,
        )
        )
    elif args.strategy == 'PGPE':
        # Update basic parameters of PGPE strategy
        es_params = strategy.default_params.replace(
            sigma_init =args.sigma_init,  # Initial scale of isotropic Gaussian noise
            sigma_decay=args.sigma_decay,  # Multiplicative decay factor
            sigma_limit=args.sigma_limit,  # Smallest possible scale
            sigma_lrate=args.sigma_lrate,  # Learning rate for scale
            sigma_max_change=args.sigma_max_change,  # clips adaptive sigma to 20%
            init_min=args.init_min,  # Range of parameter mean initialization - Min
            init_max=args.init_max,  # Range of parameter mean initialization - Max
            clip_min=args.clip_min,  # Range of parameter proposals - Min
            clip_max=args.clip_max  # Range of parameter proposals - Max
        )

        # Update optimizer-specific parameters of Adam
        es_params = es_params.replace(opt_params=es_params.opt_params.replace(
            lrate_init=args.lr_init,  # Initial learning rate
            lrate_decay=args.lrate_init,  # Multiplicative decay factor
            lrate_limit=args.lrate_limit,  # Smallest possible lrate
            beta_1=args.beta_1,  # Adam - beta_1
            beta_2=args.beta_2,  # Adam - beta_2
            eps=args.eps,  # eps constant,
        )
        )
    return strategy, es_params
import jax.numpy as jnp
from evosax import Strategies, NetworkMapper

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
    else:
        print('ERROR Task is not supported')


def get_strategy_and_params(pop_size, num_dims, args):
    MyStrategy = Strategies[args.strategy]
    # strategy = MyStrategy(popsize=pop_size, num_dims=num_dims)
    strategy = MyStrategy(popsize=pop_size, num_dims=num_dims, opt_name=args.opt_name)
    es_params = strategy.default_params
    if args.strategy == 'MahiES':
        es_params = strategy.default_params.replace(
            sigma_init=args.sigma_init,  # Initial scale of isotropic Gaussian noise
            sigma_decay=args.sigma_decay,  # Multiplicative decay factor
            sigma_limit=args.sigma_limit,  # Smallest possible scale
            mahi_lrate=args.lr,  # Learning rate
            init_min=args.init_min,  # Range of parameter mean initialization - Min
            init_max=args.init_max,  # Range of parameter mean initialization - Max
        )
    elif args.strategy == 'OpenES':
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
        es_params = es_params.replace(
            opt_params=es_params.opt_params.replace(
                lrate_init=args.lrate_init,  # Initial learning rate
                lrate_decay=args.lrate_decay,  # Multiplicative decay factor
                lrate_limit=args.lrate_limit,  # Smallest possible lrate
                beta_1=args.beta_1,  # Adam - beta_1
                beta_2=args.beta_2,  # Adam - beta_2
                eps=args.eps,  # eps constant,
                momentum=args.momentum_es,  # Momentum
            )
        )
    elif args.strategy == 'PGPE':
        # Update basic parameters of PGPE strategy
        es_params = strategy.default_params.replace(
            sigma_init=args.sigma_init,  # Initial scale of isotropic Gaussian noise
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
            lrate_init=args.lrate_init,  # Initial learning rate
            lrate_decay=args.lrate_decay,  # Multiplicative decay factor
            lrate_limit=args.lrate_limit,  # Smallest possible lrate
            beta_1=args.beta_1,  # Adam - beta_1
            beta_2=args.beta_2,  # Adam - beta_2
            eps=args.eps,  # eps constant,
        )
        )
    elif args.strategy == 'CMA_ES':
        # Update basic parameters of PGPE strategy
        mu_eff: float
        c_1: float
        c_mu: float
        c_sigma: float
        d_sigma: float
        c_c: float
        chi_n: float
        c_m: float = 1.0
        sigma_init: float = 0.065
        init_min: float = 0.0
        init_max: float = 0.0
        clip_min: float = -jnp.finfo(jnp.float32).max
        clip_max: float = jnp.finfo(jnp.float32).max

        es_params = strategy.default_params.replace(
            c_m=args.c_m,  # Range of parameter proposals - Min
            sigma_init=args.sigma_init,  # Initial scale of isotropic Gaussian noise
            init_min=args.init_min,  # Range of parameter mean initialization - Min
            init_max=args.init_max,  # Range of parameter mean initialization - Max
            clip_min=args.clip_min,  # Range of parameter proposals - Min
            clip_max=args.clip_max  # Range of parameter proposals - Max
        )

    return strategy, es_params


def get_strategy_and_params_cma(pop_size, num_dims, args):
    MyStrategy = Strategies[args.strategy]
    # strategy = MyStrategy(popsize=pop_size, num_dims=num_dims)
    strategy = MyStrategy(popsize=pop_size, num_dims=num_dims)
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
            lrate_init=args.lrate_init,  # Initial learning rate
            lrate_decay=args.lrate_decay,  # Multiplicative decay factor
            lrate_limit=args.lrate_limit,  # Smallest possible lrate
            beta_1=args.beta_1,  # Adam - beta_1
            beta_2=args.beta_2,  # Adam - beta_2
            eps=args.eps,  # eps constant,
            momentum=args.momentum_es,  # Momentum

        )
        )
    elif args.strategy == 'PGPE':
        # Update basic parameters of PGPE strategy
        es_params = strategy.default_params.replace(
            sigma_init=args.sigma_init,  # Initial scale of isotropic Gaussian noise
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
            lrate_init=args.lrate_init,  # Initial learning rate
            lrate_decay=args.lrate_decay,  # Multiplicative decay factor
            lrate_limit=args.lrate_limit,  # Smallest possible lrate
            beta_1=args.beta_1,  # Adam - beta_1
            beta_2=args.beta_2,  # Adam - beta_2
            eps=args.eps,  # eps constant,
        )
        )
    return strategy, es_params
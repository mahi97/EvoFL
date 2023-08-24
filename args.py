import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Evolutionary Federated Learning')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='JAX Random Number Generator Seed')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/Vision-MNIST/openes.yaml",
        # default="./configs/CartPole-v1/es.yaml",
        help="Path to configuration yaml.",
    )
    args = parser.parse_args()

    return args

from .mlp import MLP
from .cnn import CNN, All_CNN_C, ResNet
from .lstm import LSTM
from .hyper_networks import HyperNetworkMLP


# Helper that returns model based on string name
NetworkMapper = {
    "MLP": MLP,
    "CNN": CNN,
    "All_CNN_C": All_CNN_C,
    "LSTM": LSTM,
    "ResNet": ResNet,
}

__all__ = [
    "MLP",
    "CNN",
    "All_CNN_C",
    "LSTM",
    "NetworkMapper",
    "HyperNetworkMLP",
]

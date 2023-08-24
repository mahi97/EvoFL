from .control_brax import BraxFitness
from .control_gym import GymFitness
from .vision import VisionFitness
from .classic import ClassicFitness
from .sequence import SequenceFitness
from .fed_vision import FederatedVisionFitness
from .fed_vision_new import NewFederatedVisionFitness
from .fed_control_gym import FederatedGymFitness
from .fed_control_brax import FederatedBraxFitness
from .vision_new import VisionFitnessNew

ProblemMapper = {
    "Gym": GymFitness,
    "Brax": BraxFitness,
    "Vision": VisionFitness,
    "Classic": ClassicFitness,
    "Sequence": SequenceFitness,
    "FedVision": FederatedVisionFitness,
    "FedGym": FederatedGymFitness,
    "FedBrax": FederatedBraxFitness,
    "VisionNew": VisionFitnessNew,
}

__all__ = [
    "BraxFitness",
    "GymFitness",
    "VisionFitness",
    "ClassicFitness",
    "SequenceFitness",
    "ProblemMapper",
    "FederatedVisionFitness",
    "VisionFitnessNew"
]

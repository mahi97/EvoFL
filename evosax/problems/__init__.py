from .control_brax import BraxFitness
from .control_gym import GymFitness
from .vision import VisionFitness
from .classic import ClassicFitness
from .sequence import SequenceFitness
from .vision_new import VisionFitnessNew

ProblemMapper = {
    "Gym": GymFitness,
    "Brax": BraxFitness,
    "Vision": VisionFitness,
    "Classic": ClassicFitness,
    "Sequence": SequenceFitness,
    "VisionNew": VisionFitnessNew,
}

__all__ = [
    "BraxFitness",
    "GymFitness",
    "VisionFitness",
    "ClassicFitness",
    "SequenceFitness",
    "ProblemMapper",
    "VisionFitnessNew"
]

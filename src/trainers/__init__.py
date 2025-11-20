from src.utils.registry import TRAINER_REGISTRY
from src.trainers.classification_trainer import ClassificationTrainer

__all__ = ['build_trainer']


def build_trainer(config):
    return TRAINER_REGISTRY.get(config['trainer'])(config)

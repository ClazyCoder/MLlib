from src.utils.registry import TRAINER_REGISTRY
from src.trainers.classification_trainer import ClassificationTrainer
from src.trainers.config import TrainerConfig
from src.utils.config import RootConfig
__all__ = ['build_trainer']


def build_trainer(config: RootConfig):
    trainer_config = TrainerConfig(**config.trainer_config)
    return TRAINER_REGISTRY.get(trainer_config.name)(config)

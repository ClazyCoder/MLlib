from src.utils.registry import MODEL_REGISTRY
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from src.models.config import ModelConfig

__all__ = ['build_model']


def build_model(config: ModelConfig):
    return MODEL_REGISTRY.get(config.name)(config)

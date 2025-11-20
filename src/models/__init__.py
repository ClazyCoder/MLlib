from src.utils.registry import MODEL_REGISTRY
from src.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

__all__ = ['build_model']


def build_model(config):
    return MODEL_REGISTRY.get(config['model'])(config)

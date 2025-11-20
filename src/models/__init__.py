from utils.registry import MODEL_REGISTRY
from src.models.resnet import *

__all__ = ['build_model']


def build_model(config):
    return MODEL_REGISTRY.get(config['model'])(config)

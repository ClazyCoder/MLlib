from src.utils.registry import LOSS_REGISTRY

__all__ = ['build_loss']


def build_loss(config):
    return LOSS_REGISTRY.get(config['loss'])(config)

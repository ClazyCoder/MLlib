from src.utils.registry import LOSS_REGISTRY
from src.losses.ce_loss import CELoss

__all__ = ['build_loss']


def build_loss(config):
    return LOSS_REGISTRY.get(config['criterion'])(config)

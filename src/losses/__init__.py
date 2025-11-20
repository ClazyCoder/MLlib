from src.utils.registry import LOSS_REGISTRY
from src.losses.ce_loss import CELoss
from src.losses.config import LossConfig

__all__ = ['build_loss']


def build_loss(config: LossConfig):
    return LOSS_REGISTRY.get(config.name)(config)

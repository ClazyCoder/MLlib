import torch.nn as nn
import torch.nn.functional as F
from src.losses.config import LossConfig
from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CELoss(nn.Module):
    def __init__(self, config: LossConfig):
        super(CELoss, self).__init__()
        self.weight = config.weight
        self.ignore_index = config.ignore_index

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index)

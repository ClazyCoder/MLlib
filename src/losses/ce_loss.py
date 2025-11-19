import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CELoss(nn.Module):
    def __init__(self, config):
        super(CELoss, self).__init__()
        self.weight = config['loss_config'].get('weight', None)
        self.ignore_index = config['loss_config'].get('ignore_index', -100)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index)

import torch.nn as nn
from src.models.config import ModelConfig
import torch


class NextTokenPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super(NextTokenPredictor, self).__init__()
        self.config = config
        # TODO : Implement Model Architecture

    def forward(self, x):
        # TODO : Implement Forward Pass
        pass

    def generate(self, x):
        pass
        # TODO : Implement Autoregressive Generation

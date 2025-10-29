from src.trainers.base_trainer import BaseTrainer
from utils.device import move_to_device, get_device
from src.models import build_model
from src.datasets import build_dataset
from src.losses import build_loss
import torch
from src.utils.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register()
class ClassificationTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassificationTrainer, self).__init__(config)

        self.model = build_model(config.get('model', None))
        self.criterion = build_loss(config.get('criterion', None))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.get('lr', 0.001))
        self.dataloader = build_dataset(config.get('dataloader', None))
        self.device = get_device()

    def train(self):
        self.model.train()
        total_loss = 0
        for images, labels in self.dataloader:
            images = move_to_device(images, self.device)
            labels = move_to_device(labels, self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validation(self):
        pass

    def test(self):
        pass

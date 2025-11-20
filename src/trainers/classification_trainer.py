from src.trainers.base_trainer import BaseTrainer
from src.utils.device import move_to_device, get_device
from src.models import build_model
from src.datasets import build_dataset
from src.losses import build_loss
from src.metrics import build_metric
import torch
from torch.utils.data import DataLoader
import os
from src.utils.registry import TRAINER_REGISTRY
from logging import getLogger


@TRAINER_REGISTRY.register()
class ClassificationTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassificationTrainer, self).__init__(config)
        self.config = config
        self.model = build_model(config)
        self.criterion = build_loss(self.config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.get('lr', 0.001))
        self.train_dataloader = DataLoader(build_dataset(
            self.config, 'train'), batch_size=self.config.get('batch_size', 16), shuffle=True)
        self.val_dataloader = DataLoader(build_dataset(
            self.config, 'val'), batch_size=self.config.get('batch_size', 16), shuffle=False)
        self.device = get_device()
        self.best_val_accuracy = 0
        self.metric = build_metric(self.config)

    def train(self):
        logger = getLogger(__name__)
        logger.info(f"Training started.")
        total_train_loss = 0
        total_train_accuracy = 0
        total_val_loss = 0
        total_val_accuracy = 0
        move_to_device(self.model, self.device)
        for epoch in range(self.config.get('epochs', 10)):
            self.model.train()
            logger.info(f"Epoch {epoch} started.")
            for images, labels in self.train_dataloader:
                images = move_to_device(images, self.device)
                labels = move_to_device(labels, self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_train_loss += loss.item()
                total_train_accuracy += self.metric.compute(outputs, labels)
                loss.backward()
                self.optimizer.step()
            logger.info(
                f"Train Loss: {total_train_loss / len(self.train_dataloader)}")
            logger.info(
                f"Train Accuracy: {total_train_accuracy / len(self.train_dataloader)}")
            self.model.eval()
            logger.info(f"Epoch {epoch} completed successfully.")
            for images, labels in self.val_dataloader:
                images = move_to_device(images, self.device)
                labels = move_to_device(labels, self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                total_val_accuracy += self.metric.compute(outputs, labels)
            logger.info(
                f"Val Loss: {total_val_loss / len(self.val_dataloader)}")
            logger.info(
                f"Val Accuracy: {total_val_accuracy / len(self.val_dataloader)}")
            if total_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = total_val_accuracy
                torch.save(self.model.state_dict(), os.path.join(
                    self.config.get('model_config').get('save_dir', ''), 'best_model.pth'))
                logger.info(
                    f"Best validation accuracy updated to {self.best_val_accuracy} at epoch {epoch}")
            logger.info(f"Epoch {epoch} completed successfully.")
        logger.info(f"Training completed successfully.")

    def test(self):
        logger = getLogger(__name__)
        logger.info(f"Testing started.")
        move_to_device(self.model, self.device)
        total_test_loss = 0
        total_test_accuracy = 0
        self.model.eval()
        for images, labels in self.test_dataloader:
            images = move_to_device(images, self.device)
            labels = move_to_device(labels, self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_test_loss += loss.item()
            total_test_accuracy += self.metric.compute(outputs, labels)
        logger.info(
            f"Test Loss: {total_test_loss / len(self.test_dataloader)}")
        logger.info(
            f"Test Accuracy: {total_test_accuracy / len(self.test_dataloader)}")
        logger.info(f"Testing completed successfully.")

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
from src.utils.config import RootConfig
from src.models.config import ModelConfig
from src.trainers.config import TrainerConfig
from src.datasets.config import DatasetConfig
from src.losses.config import LossConfig
from src.metrics.config import MetricConfig


@TRAINER_REGISTRY.register()
class ClassificationTrainer(BaseTrainer):
    def __init__(self, config: RootConfig):
        trainer_config = TrainerConfig(**config.trainer_cfg)
        super(ClassificationTrainer, self).__init__(trainer_config)
        self.config = trainer_config
        model_config = ModelConfig(**config.model_cfg)
        self.model = build_model(model_config)
        loss_config = LossConfig(**config.loss_cfg)
        self.loss = build_loss(loss_config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr)

        dataset_config = DatasetConfig(**config.dataset_cfg)
        self.train_dataloader = DataLoader(build_dataset(
            dataset_config, 'train'), batch_size=self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(build_dataset(
            dataset_config, 'val'), batch_size=self.config.batch_size, shuffle=False)
        if dataset_config.test_dataset_path:
            self.test_dataloader = DataLoader(build_dataset(
                dataset_config, 'test'), batch_size=self.config.batch_size, shuffle=False)
        else:
            self.test_dataloader = self.val_dataloader
        self.device = get_device()
        self.best_val_accuracy = 0
        metric_config_dict = dict(config.metric_cfg)
        if 'num_classes' not in metric_config_dict:
            metric_config_dict['num_classes'] = model_config.num_classes
        metric_config = MetricConfig(**metric_config_dict)
        self.metric = build_metric(metric_config)

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

    def train(self):
        logger = getLogger(__name__)
        logger.info(f"Training started.")
        move_to_device(self.model, self.device)
        for epoch in range(self.config.epochs):
            total_train_loss = 0
            total_train_accuracy = 0
            total_val_loss = 0
            total_val_accuracy = 0
            self.model.train()
            logger.info(f"Epoch {epoch} started.")

            for images, labels in self.train_dataloader:
                images = move_to_device(images, self.device)
                labels = move_to_device(labels, self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                total_train_loss += loss.item()
                total_train_accuracy += self.metric.compute(outputs, labels)
                loss.backward()
                self.optimizer.step()
            logger.info(
                f"Train Loss: {total_train_loss / len(self.train_dataloader)}")
            logger.info(
                f"Train Accuracy: {total_train_accuracy / len(self.train_dataloader)}")
            self.model.eval()

            with torch.no_grad():
                for images, labels in self.val_dataloader:
                    images = move_to_device(images, self.device)
                    labels = move_to_device(labels, self.device)
                    outputs = self.model(images)
                    loss = self.loss(outputs, labels)
                    total_val_loss += loss.item()
                    total_val_accuracy += self.metric.compute(outputs, labels)
                logger.info(
                    f"Val Loss: {total_val_loss / len(self.val_dataloader)}")
                logger.info(
                    f"Val Accuracy: {total_val_accuracy / len(self.val_dataloader)}")
                if len(self.val_dataloader) > 0:
                    avg_val_accuracy = total_val_accuracy / len(self.val_dataloader)
                    if avg_val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = avg_val_accuracy
                        torch.save(self.model.state_dict(), os.path.join(
                            self.config.save_dir, 'best_model.pth'))
                        logger.info(
                            f"Best validation accuracy updated to {self.best_val_accuracy} at epoch {epoch}")
            logger.info(f"Epoch {epoch} completed successfully.")
        logger.info(f"Training completed successfully.")

    def test(self):
        logger = getLogger(__name__)
        logger.info(f"Testing started.")
        # Move model to device only if not already on device
        if next(self.model.parameters()).device != self.device:
            move_to_device(self.model, self.device)
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images = move_to_device(images, self.device)
                labels = move_to_device(labels, self.device)
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                total_test_loss += loss.item()
                total_test_accuracy += self.metric.compute(outputs, labels)
        logger.info(
            f"Test Loss: {total_test_loss / len(self.test_dataloader)}")
        logger.info(
            f"Test Accuracy: {total_test_accuracy / len(self.test_dataloader)}")
        logger.info(f"Testing completed successfully.")

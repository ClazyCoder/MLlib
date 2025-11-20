from src.utils.registry import DATASET_REGISTRY
from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.config import DatasetConfig


__all__ = ['build_dataset']


def build_dataset(config: DatasetConfig, split: str):
    return DATASET_REGISTRY.get(config.name)(config, split)

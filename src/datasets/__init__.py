from src.utils.registry import DATASET_REGISTRY
from src.datasets.classification_dataset import ClassificationDataset


__all__ = ['build_dataset']


def build_dataset(config, type):
    return DATASET_REGISTRY.get(config['dataset'])(config, type)

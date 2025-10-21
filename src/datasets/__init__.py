from src.utils.registry import DATASET_REGISTRY


__all__ = ['build_dataset']


def build_dataset(config):
    return DATASET_REGISTRY.get(config['dataset'])(config)

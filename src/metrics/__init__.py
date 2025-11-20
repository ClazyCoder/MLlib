from utils.registry import METRIC_REGISTRY
from src.metrics.accuracy import Accuracy

__all__ = ['build_metric']


def build_metric(config):
    return METRIC_REGISTRY.get(config['metric'])(config)

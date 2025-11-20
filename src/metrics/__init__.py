from src.utils.registry import METRIC_REGISTRY
from src.metrics.accuracy import Accuracy
from src.metrics.config import MetricConfig

__all__ = ['build_metric']


def build_metric(config: MetricConfig):
    return METRIC_REGISTRY.get(config.name)(config)

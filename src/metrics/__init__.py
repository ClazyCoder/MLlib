from utils.registry import METRIC_REGISTRY

__all__ = ['build_metric']

def build_metric(config):
    return METRIC_REGISTRY.get(config['metric'])(config)
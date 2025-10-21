from utils.registry import TRAINER_REGISTRY
__all__ = ['build_trainer']
def build_trainer(config):
    return TRAINER_REGISTRY.get(config['trainer'])(config)
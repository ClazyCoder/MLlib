from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

def build_model(config):
    return MODEL_REGISTRY.get(config['model'])(config)

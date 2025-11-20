class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, obj=None):
        # If obj is None, it is a decorator
        if obj is None:
            def decorator(func_or_cls):
                # name is the name of the function or class
                name = func_or_cls.__name__
                self._register(name, func_or_cls)
                return func_or_cls
            return decorator

        # name is the name of the object
        name = obj.__name__
        self._register(name, obj)

    def _register(self, name, obj):
        assert (
            name not in self._registry), f"{name} already registered in {self._name}"
        self._registry[name] = obj

    def get(self, name):
        obj = self._registry.get(name)
        if obj is None:
            error_msg = f"{name} not found in {self._name} registry\n"
            error_msg += f"{self._name} registry contains the following items:\n"
            error_msg += f"{self}"
            raise KeyError(error_msg)
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry.items())

    def __repr__(self):
        repr_str = f"{self._name} registry containing the following items:\n"
        for name, obj in self._registry.items():
            repr_str += f"  - {name}: {obj.__class__.__name__}\n"
        return repr_str


MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
TRAINER_REGISTRY = Registry("trainer")
DATASET_REGISTRY = Registry("dataset")
METRIC_REGISTRY = Registry("metric")

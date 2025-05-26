class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, name_or_cls, cls=None):
        if cls is None:
            name = name_or_cls

            def decorator(func_or_cls):
                self._register(name, func_or_cls)
                return func_or_cls
            return decorator
        else:
            name = name_or_cls
            self._register(name, cls)

    def _register(self, name, obj):
        if name in self._registry:
            raise KeyError(f"{name} already registered in {self._name}")
        self._registry[name] = obj

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name]

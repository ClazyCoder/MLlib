class Registry:
    def __init__(self, name):
        self._name = name
        self._registry = {}

    def register(self, obj=None):
        if obj is None:
            def decorator(func_or_cls):
                name = func_or_cls.__name__.lower()
                self._register(name, func_or_cls)
                return func_or_cls
            return decorator

        name = obj.__name__.lower()
        self._register(name, obj)

    def _register(self, name, obj):
        assert (
            name not in self._registry), f"{name} already registered in {self._name}"
        self._registry[name] = obj

    def get(self, name):
        obj = self._registry.get(name)
        if obj is None:
            raise KeyError(f"{name} not found in {self._name}")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry.items())

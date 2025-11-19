import yaml


class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    def __iter__(self):
        return iter(self.config.items())

    def __len__(self):
        return len(self.config)

    def save(self, config_path: str):
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    def update(self, config: dict):
        self.config.update(config)

    def keys(self):
        return self.config.keys()

    def values(self):
        return self.config.values()

    def items(self):
        return self.config.items()

    def load(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def get(self, key, default=None):
        # TODO : check validity of value(type, range, etc.)
        return self.config.get(key, default)

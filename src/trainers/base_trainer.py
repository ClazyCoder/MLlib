from abc import *


class BaseTrainer(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validation(self):
        pass

    @abstractmethod
    def test(self):
        pass

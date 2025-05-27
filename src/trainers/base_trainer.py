from abc import *


class BaseTrainer(ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validation(self):
        pass

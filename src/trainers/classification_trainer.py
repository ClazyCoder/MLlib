from src.trainers.base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassificationTrainer, self).__init__(config)

    def train(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass

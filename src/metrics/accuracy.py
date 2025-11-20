from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class Accuracy:
    def __init__(self, config):
        self.num_classes = config.get('model_config').get('num_classes', 2)
        self.total_samples = 0
        self.correct_samples = 0

    def compute(self, outputs, targets):
        if self.num_classes == 1:
            preds = (outputs > 0.5).long()
        else:
            _, preds = outputs.max(1)
        correct = preds.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        self.total_samples += total
        self.correct_samples += correct
        return accuracy

    def reset(self):
        self.total_samples = 0
        self.correct_samples = 0

    def result(self):
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

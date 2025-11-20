import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)
from src.utils.registry import MODEL_REGISTRY


class BaseResNet(nn.Module):
    def __init__(self, model_name, num_classes=1000, pretrained=True):
        """
        ResNet 모델의 기본 클래스

        Args:
            model_name (str): 모델 이름 ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            num_classes (int): 분류할 클래스의 수
            pretrained (bool): 사전 학습된 가중치 사용 여부
        """
        super(BaseResNet, self).__init__()

        # 모델과 가중치 매핑
        model_weights = {
            'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),
            'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1),
            'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V2),
            'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V2),
            'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V2)
        }

        if model_name not in model_weights:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

        model_fn, weights = model_weights[model_name]

        if pretrained:
            self.model = model_fn(weights=weights)
        else:
            self.model = model_fn(weights=None)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


@MODEL_REGISTRY.register()
class ResNet18(BaseResNet):
    def __init__(self, config=None):
        if config is not None:
            num_classes = config.get('model_config').get(
                'num_classes', 1000)
            pretrained = config.get('model_config').get(
                'pretrained', True)
        super(ResNet18, self).__init__('resnet18', num_classes, pretrained)


@MODEL_REGISTRY.register()
class ResNet34(BaseResNet):
    def __init__(self, config=None):
        if config is not None:
            num_classes = config.get('model_config').get(
                'num_classes', 1000)
            pretrained = config.get('model_config').get(
                'pretrained', True)
        super(ResNet34, self).__init__('resnet34', num_classes, pretrained)


@MODEL_REGISTRY.register()
class ResNet50(BaseResNet):
    def __init__(self, config=None):
        if config is not None:
            num_classes = config.get('model_config').get(
                'num_classes', 1000)
            pretrained = config.get('model_config').get(
                'pretrained', True)
        super(ResNet50, self).__init__('resnet50', num_classes, pretrained)


@MODEL_REGISTRY.register()
class ResNet101(BaseResNet):
    def __init__(self, config=None):
        if config is not None:
            num_classes = config.get('model_config').get(
                'num_classes', 1000)
            pretrained = config.get('model_config').get(
                'pretrained', True)
        super(ResNet101, self).__init__('resnet101', num_classes, pretrained)


@MODEL_REGISTRY.register()
class ResNet152(BaseResNet):
    def __init__(self, config=None):
        if config is not None:
            num_classes = config.get('model_config').get(
                'num_classes', 1000)
            pretrained = config.get('model_config').get(
                'pretrained', True)
        super(ResNet152, self).__init__('resnet152', num_classes, pretrained)

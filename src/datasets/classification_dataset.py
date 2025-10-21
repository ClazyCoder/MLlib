from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch
from src.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.Resize((224, 224))(image)
            image = torchvision.transforms.ToTensor()(image)
        return image, self.labels[idx]

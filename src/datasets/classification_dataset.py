from torch.utils.data import Dataset
from PIL import Image
import torchvision
import json
from src.utils.registry import DATASET_REGISTRY
from src.datasets.config import DatasetConfig


@DATASET_REGISTRY.register()
class ClassificationDataset(Dataset):
    def __init__(self, config: DatasetConfig, type: str = 'train'):
        path_map = {
            'train': config.train_dataset_path,
            'val': config.val_dataset_path,
            'test': config.test_dataset_path
        }
        with open(path_map[type], 'r') as f:
            data = json.load(f)
        self.image_paths = data['image_paths']
        self.labels = data['labels']
        self.transform = torchvision.transforms.ToTensor()  # TODO : add transform builder

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

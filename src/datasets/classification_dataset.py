from torch.utils.data import Dataset
from PIL import Image
import torchvision
import json
from src.utils.registry import DATASET_REGISTRY
from src.datasets.config import DatasetConfig


@DATASET_REGISTRY.register()
class ClassificationDataset(Dataset):
    def __init__(self, config: DatasetConfig, split: str = 'train'):
        path_map = {
            'train': config.train_dataset_path,
            'val': config.val_dataset_path,
            'test': config.test_dataset_path
        }
        if split not in path_map:
            raise ValueError(f"Unknown dataset split '{split}'. Must be one of {list(path_map.keys())}.")
        dataset_path = path_map[split]
        if dataset_path is None:
            raise ValueError(f"Dataset path for '{split}' is not configured.")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        self.image_paths = data['image_paths']
        self.labels = data['labels']
        self.transform = torchvision.transforms.ToTensor()  # TODO : add transform builder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            if self.transform:
                image = self.transform(image)
            else:
                image = torchvision.transforms.Resize((224, 224))(image)
                image = torchvision.transforms.ToTensor()(image)
        return image, self.labels[idx]

from torch.utils.data import Dataset
from PIL import Image
import torchvision
import json
from src.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ClassificationDataset(Dataset):
    def __init__(self, config):
        with open(config.get('data_path', None), 'r') as f:
            data = json.load(f)
        self.image_paths = data.get('image_paths', None)
        self.labels = data.get('labels', None)
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

from pydantic import BaseModel, ConfigDict
from typing import Optional


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: Optional[str] = None

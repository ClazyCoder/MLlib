from pydantic import BaseModel, ConfigDict
from typing import Dict, Any


class RootConfig(BaseModel):
    model_config: Dict[str, Any]
    loss_config: Dict[str, Any]
    trainer_config: Dict[str, Any]
    metric_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    model_config = ConfigDict(extra="forbid")

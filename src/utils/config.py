from pydantic import BaseModel, ConfigDict
from typing import Dict, Any


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_cfg: Dict[str, Any]
    loss_cfg: Dict[str, Any]
    trainer_cfg: Dict[str, Any]
    metric_cfg: Dict[str, Any]
    dataset_cfg: Dict[str, Any]

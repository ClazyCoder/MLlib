from pydantic import BaseModel, ConfigDict
from typing import Optional


class MetricConfig(BaseModel):
    name: str
    weight: Optional[float] = None
    ignore_index: int = -100
    model_config = ConfigDict(extra="allow")

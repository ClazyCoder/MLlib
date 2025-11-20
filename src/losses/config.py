from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    weight: Optional[Any] = None
    ignore_index: int = -100

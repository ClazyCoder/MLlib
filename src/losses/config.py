from pydantic import BaseModel, ConfigDict
from typing import Any, Optional

class LossConfig(BaseModel):
    name: str
    weight: Optional[Any] = None
    ignore_index: int = -100
    model_config = ConfigDict(extra="allow")

from pydantic import BaseModel, ConfigDict


class LossConfig(BaseModel):
    name: str
    model_config = ConfigDict(extra="allow")

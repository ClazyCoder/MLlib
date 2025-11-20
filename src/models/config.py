from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    num_classes: int = 1000
    pretrained: bool = True

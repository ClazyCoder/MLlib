from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    name: str
    num_classes: int = 1000
    pretrained: bool = True
    model_config = ConfigDict(extra="allow")

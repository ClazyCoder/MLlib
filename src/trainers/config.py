from pydantic import BaseModel, ConfigDict


class TrainerConfig(BaseModel):
    name: str
    lr: float = 0.001
    batch_size: int = 16
    epochs: int = 10
    model_config = ConfigDict(extra="allow")

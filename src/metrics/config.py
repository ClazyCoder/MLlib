from pydantic import BaseModel, ConfigDict


class MetricConfig(BaseModel):
    name: str
    model_config = ConfigDict(extra="allow")

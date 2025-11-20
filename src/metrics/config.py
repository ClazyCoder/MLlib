from pydantic import BaseModel, ConfigDict


class MetricConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str

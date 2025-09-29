from typing import Any

from pydantic import BaseModel

training_strategy_registry = {}


class TrainingStrategyConfig(BaseModel):
    name: str

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name is None:
            raise ValueError(f"Default value of 'name' shall be set for training strategy config class:{cls}")

        training_strategy_registry[name] = cls

    class Config:
        extra = "allow"


class OneGpuTrainingStrategyConfig(TrainingStrategyConfig):
    name: str = 'one_gpu'

class AccelerateTrainingStrategyConfig(TrainingStrategyConfig):
    name: str = 'accelerate'



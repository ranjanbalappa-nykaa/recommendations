import enum
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel
from typing import Any, Generic, List, Optional, TypeVar

from commons.configs.feature_config import FeaturesConfig, Task
from commons.pipeline.model_builder import ModelBuilder
from commons.pipeline.types import Model, Stats


class ModelKind(str, enum.Enum):
    RANKER = "ranker"
    CROSSDOMAIN = "LTHM"


model_registry = {}

class ModelConfig(BaseModel, Generic[Model], ABC):
    kind: ModelKind
    type: str
    name: str
    version: str
    features: FeaturesConfig
    tasks: Optional[List[Task]] = None

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = getattr(cls, "kind", None)
        if kind is None:
            raise ValueError(f"Default value of 'kind' shall be set for model config class:{cls}")
        name = getattr(cls, "name", None)
        if name is None:
            raise ValueError(f"Default value of 'name' shall be set for model config class:{cls}")

        model_registry[f"{kind.value}/{name}"] = cls

    @abstractmethod
    def get_builder(self) -> ModelBuilder[Model]:
        pass

    def custom_data_preprocessor(self, df: pd.DataFrame, kind: str = "train") -> pd.DataFrame:
        return df
    
    def special_data_prepreprocessor(self, df: pd.DataFrame, kind: str = "train") -> pd.DataFrame:
        return df

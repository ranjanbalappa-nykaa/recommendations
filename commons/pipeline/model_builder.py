from typing import Generic, Optional

from commons.pipeline.types import Model, Stats
from abc import ABC, abstractmethod


class ModelBuilder(Generic[Model], ABC):
    def __init__(self, stats: Optional[Stats]):
        self.stats = stats

    @abstractmethod
    def build(self) -> Model:
        pass
from abc import ABC, abstractmethod
from typing import Generic, List, Optional

from commons.configs.trainer_pipeline_config import TrainerPipelineConfig
from commons.pipeline.model_builder import ModelBuilder
from commons.pipeline.model_checkpointer import ModelCheckpointer
from commons.pipeline.types import Model
from commons.data.data_loader_strategy import DataLoaderStrategy


class TrainingStrategy(Generic[Model], ABC):
    @abstractmethod
    def train(
            self,
            model_builder: ModelBuilder[Model],
            data_loader_strategy: DataLoaderStrategy,
            train_data_paths: List[str],
            val_data_paths: List[str],
            pipeline_config: TrainerPipelineConfig,
            model_checkpointer: Optional[ModelCheckpointer]
    ) -> Model:
        raise NotImplementedError("Subclasses must implement this method")
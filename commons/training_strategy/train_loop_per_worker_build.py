from typing import Any, List, Optional, Dict, Generator, Tuple, Callable

from ray.air import session
import torch
from torch.utils.data import DataLoader

from commons.configs.trainer_pipeline_config import TrainerPipelineConfig
from commons.pipeline.model_checkpointer import ModelCheckpointer
from commons.data.dataset_generator_utils import get_paths_for_worker
from commons.pipeline.model_builder import ModelBuilder
from commons.data import get_torch_dataloader
from commons.data.data_loader_strategy import DataLoaderStrategy



class TrainLoopPerWorkerBuilder:
    def __init__(
            self,
            model_builder: ModelBuilder,
            data_loader_strategy: DataLoaderStrategy,
            pipeline_config: TrainerPipelineConfig
    ):
        self.model_builder = model_builder
        self.data_loader_strategy = data_loader_strategy
        self.pipeline_config = pipeline_config

    def _data_for_worker(
            self,
            kind: str,
            worker_id: int,
            data_paths: Optional[List[str]],
            seed: Optional[int] = None
    ) -> Optional[DataLoader]:
        if data_paths is None or len(data_paths) <= 0:
            return None

        model_train_config = self.pipeline_config.train
        model_val_config = self.pipeline_config.eval
        batch_size = model_train_config.batch_size if kind == "train" else model_val_config.eval_batch_size
        num_workers = model_train_config.num_workers if kind == "train" else model_val_config.num_workers
        num_steps = model_train_config.train_steps if kind == "train" else model_train_config.validation_steps

        paths = get_paths_for_worker(worker_id=worker_id, data_paths=data_paths, num_workers=num_workers, seed=seed)

        return get_torch_dataloader(
            kind=kind,
            worker_id=worker_id,
            paths=paths,
            batch_size=batch_size,
            num_steps=num_steps,
            data_loader_strategy=self.data_loader_strategy,
            model_config=self.pipeline_config.model,
            fs_config=self.pipeline_config.dataset.filesystem_config,
        )

    def per_loop_fit(
            self, rank: int,
            per_epoch_dataset_generator: Callable[
                [],
                Generator[Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]], Any, None]
            ],
            config: Optional[Dict],
            model_checkpointer: Optional[ModelCheckpointer] = None):
        raise ValueError("Subclasses must implement this method")

    def build(
            self,
            train_data_paths: List[str],
            val_data_paths: Optional[List[str]] = None,
            per_loop_fit_config: Optional[Dict] = None,
            model_checkpointer: Optional[ModelCheckpointer] = None):
        model_train_config = self.pipeline_config.train

        def train_loop_per_worker():
            worker_id = session.get_world_rank()

            def per_epoch_dataset_generator():
                for epoch in range(model_train_config.epochs):
                    train_data = self._data_for_worker("train", worker_id, train_data_paths, seed=epoch)
                    val_data = self._data_for_worker("val", worker_id, val_data_paths, seed=epoch)
                    yield train_data, val_data

            self.per_loop_fit(worker_id, per_epoch_dataset_generator, per_loop_fit_config, model_checkpointer)

        return train_loop_per_worker
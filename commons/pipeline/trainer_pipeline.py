import os
import tempfile
import json
import os.path

import numpy as np
import ray
import torch

from pathlib import Path
from typing import List, Optional, Any, Dict
import logging

import pandas as pd

from commons.pipeline.types import Model
from dataclasses import dataclass

from commons.configs.trainer_pipeline_config import TrainerPipelineConfig
from commons.pipeline.model_builder import ModelBuilder
from commons.configs.feature_config import Feature
from commons.base_model_wrapper import BaseModelWrapper
from commons.data.data_loader_strategy import DataLoaderStrategy
from commons.training_strategy.training_strategy import TrainingStrategy
from commons.configs.model_config import ModelKind
from commons.data import get_torch_dataloader
from commons.pipeline.model_checkpointer import ModelCheckpointer
from commons.data.dataset_generator_utils import get_train_data_paths, get_val_data_paths
from commons.data.data_store import DataStoreAccessor, DataStoreInterface


logger = logging.getLogger(__name__)



@dataclass
class EvalResult:
    result_df: Optional[pd.DataFrame] = None
    score_df: Optional[pd.DataFrame] = None
    knn_eval_result: Optional[pd.DataFrame] = None


class TrainerPipeline:
    """ 
    Initialize TrainerPipeline with a training_strategy and data_loader
    """

    def __init__(
            self,
            pipeline_config: TrainerPipelineConfig,
            model_builder: ModelBuilder[BaseModelWrapper],
            training_strategy: TrainingStrategy[BaseModelWrapper],
            data_loader_strategy: DataLoaderStrategy,
    ):
        self.pipeline_config = pipeline_config
        self.model_builder = model_builder
        self.training_strategy = training_strategy
        self.data_loader_strategy = data_loader_strategy

        self.model_checkpointer = ModelCheckpointer(
            lambda state_dict, result_df, result_extra_day_df: self.export_model(
                state_dict=state_dict,
                eval_result=EvalResult(result_df=result_df, result_extra_day_df=result_extra_day_df),
                inference_result=None,
                training_done=False,
            )
        )
        train_paths = self._get_train_data_paths()
        if pipeline_config.export.trace:
            trace_batch = next(iter(
                get_torch_dataloader(
                    kind="train",
                    worker_id=0,
                    paths=train_paths,
                    batch_size=pipeline_config.data_loader.mini_batch_size,
                    num_steps=1,
                    data_loader_strategy=data_loader_strategy,
                    model_config=pipeline_config.model,
                    fs_config=pipeline_config.dataset.filesystem_config,
                )
            ))
            trace_batch = {
                key: trace_batch[key][:32] for key in trace_batch if isinstance(trace_batch[key], torch.Tensor)
            }
            self.trace_batch = {
                key: trace_batch[key].unsqueeze(1) if len(trace_batch[key].shape) == 1 else trace_batch[key]
                for key in trace_batch
            }
        else:
            self.trace_batch = None

    def _get_train_data_paths(self):
        return get_train_data_paths(self.pipeline_config.dataset)

    def _get_val_data_paths(self) -> List[str]:
        return get_val_data_paths(self.pipeline_config.dataset)
    
    def _get_extra_day_val_data_paths(self) -> List[str]:
        return get_val_data_paths(self.pipeline_config.dataset, for_extra_day=True)

    def execute(self):
        logger.info("Starting trainer pipeline execute")

        print(self.pipeline_config.trackers)

        # initialise trackers
        self.pipeline_config.trackers.start_run()
        self.pipeline_config.trackers.log_params_flatten('dataset', self.pipeline_config.dataset.dict())
        self.pipeline_config.trackers.log_params_flatten('train', self.pipeline_config.train.dict())
        self.pipeline_config.trackers.log_params_flatten('inference', self.pipeline_config.inference.dict())
        self.pipeline_config.trackers.log_params_flatten('eval', self.pipeline_config.eval.dict())
        self.pipeline_config.trackers.log_params_flatten('export', self.pipeline_config.export.dict())
        self.pipeline_config.trackers.log_params_flatten('training_strategy', self.pipeline_config.training_strategy.dict())
        self.pipeline_config.trackers.log_params_flatten('data_loader', self.pipeline_config.data_loader.dict())
        # self.pipeline_config.trackers.log_params_flatten('stats', self.pipeline_config.stats.dict())
        self.pipeline_config.trackers.log_params_flatten('trackers', self.pipeline_config.trackers.dict())

        model_version_dict = {"model_version": self.pipeline_config.model_version}
        self.pipeline_config.trackers.log_params(model_version_dict)

        train_data_paths = self._get_train_data_paths()
        val_data_paths = self._get_val_data_paths()

        if not self.pipeline_config.train.skip_train:
            logger.info("Starting trainer pipeline train")
            model = self.train(train_data_paths, val_data_paths)
            logger.info("Finished trainer pipeline train")
            self.export_model(
                state_dict=model.state_dict(),
                eval_result=None,
                inference_result=None,
                training_done=True,
            )
        else:
            logger.info("Skipping model export as skip_train is True")
            model = self.model_builder.build()


        if self.pipeline_config.eval is not None:
            logger.info("Starting trainer pipeline eval")
            eval_result = self.eval_model(model=model)
            logger.info("Finished trainer pipeline eval")
            self.export_model(
                state_dict=None,
                eval_result=eval_result,
                training_done=True,
            )
        else:
            logger.info("Model eval config is None, skipping eval ")

    def train(
            self,
            train_data_paths: List[str],
            val_data_paths: List[str],
    ) -> Model:
        return self.training_strategy.train(
            self.model_builder,
            self.data_loader_strategy,
            train_data_paths,
            val_data_paths,
            self.pipeline_config,
            self.model_checkpointer
        )

    def eval_model(self, model: BaseModelWrapper) -> Optional[EvalResult]:
        pass


    def export_model(
            self,
            state_dict: Optional[Any],
            eval_result: Optional[EvalResult],
            training_done: bool = False,
    ):
        """
        Saves the inference model (torch-scripted).
        If result_df / score_df is not None:
        - saves result_df score_df to a csv file
        - initializes HNSW index with train item DF and saves the index to S3
        - additionally saves a Dict with a mapping from item_id to hnsw index
        """
        if self.pipeline_config.export is None:
            return

        export_data_store = DataStoreAccessor.get_instance(self.pipeline_config.export.filesystem_config)

        with tempfile.TemporaryDirectory() as tmp:
            if eval_result is not None:
                if eval_result.result_df is not None:
                    path = os.path.join(tmp, "results.csv")
                    eval_result.result_df.to_csv(path, index=False)
                    logger.info(f"Metrics DF saved to {path}")
                if eval_result.result_extra_day_df is not None:
                    path = os.path.join(tmp, "results_extra_day.csv")
                    eval_result.result_extra_day_df.to_csv(path, index=False)
                    logger.info(f"Extra day metrics DF saved to {path}")

            if state_dict is not None:
                model = self.model_builder.build()
                model.load_state_dict(state_dict)
                model.eval()
                for i, model_ in enumerate(model.inference_models(self.trace_batch)):
                    path = os.path.join(tmp, f"model_scripted_{i}.pt")
                    torch.jit.save(
                        model_,
                        path
                    )
                    logger.info(f"Model saved to {path}")
       
            self.upload_model_artifacts(export_data_store, tmp)
            

    def upload_model_artifacts(self, export_data_store: DataStoreInterface, local_directory: str):
        self.save_model_inference_metadata(local_directory)
        export_data_store.upload_dir_recursive(
            local_directory=local_directory,
            folder=f"{self.pipeline_config.export.path_prefix}/{self.pipeline_config.model_version}"
        )
        self.pipeline_config.trackers.log_artifacts(local_directory)



    def __del__(self):
        print(self.pipeline_config.trackers)
        self.pipeline_config.trackers.end_run()
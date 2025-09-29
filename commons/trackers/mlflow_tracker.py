from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import mlflow
from mlflow.entities import RunStatus
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from commons.trackers.base import Tracker, trackers_registry

logging.basicConfig(
    format='%(asctime)s, %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:::%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

class MlflowTracker(Tracker):
    name: str = "MLFLOW"
    # mlflow_server_uri: str
    mlflow_run_tags: Dict = {}
    mlflow_artifact_location: Optional[str] = None

    def start_run(self, trackers):
        if not self.enabled:
            return

        #not required to be set in databricks
        #mlflow.set_tracking_uri(self.mlflow_server_uri)
        experiment_name = trackers.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            artifact_location = self.mlflow_artifact_location
            if artifact_location is None:
                raise ValueError("mlflow_artifact_location is None")
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=self.mlflow_artifact_location)

        run_info = mlflow.start_run(
            run_id=self.run_id,
            experiment_id=experiment_id,
            run_name=trackers.run_name,
            description=trackers.run_description)

        if self.run_id is None:
            mlflow.set_tags(self.mlflow_run_tags)
            run_id = run_info.info.run_id
            self.log_params({"run_id": run_id})
            logger.info(f"Started {self.name} RUN ID: {run_id} RUN Info: {run_info}")

            self.run_id = run_id
        else:
            logger.info(f"Resumed {self.name} RUN ID: {self.run_id} RUN Info: {run_info}")

    def end_run(self, error: bool = False):
        if not self.enabled:
            return

        status = RunStatus.FAILED if error else RunStatus.FINISHED
        mlflow.end_run(RunStatus.to_string(status))

    def log_params_flatten(self, parent_key: str, params: Dict[str, Any]):
        if not self.enabled:
            return

        for key, value in params.items():
            mlflow.log_param(f"{parent_key}_{key}", value)

    def log_params(self, params: Dict[str, Any]):
        if not self.enabled:
            return

        mlflow.log_params(params)

    def log_artifacts(self, local_directory: str):
        if not self.enabled:
            return

        mlflow.log_artifacts(local_directory)
        run = mlflow.active_run()
        experiment = mlflow.get_experiment(run.info.experiment_id)
        logger.info("mlflow experiment name: {}".format(experiment.name))
        logger.info("mlflow experiment_id: {}".format(experiment.experiment_id))
        logger.info("mlflow artifact Location: {}".format(experiment.artifact_location))

    def log_metrics(self, metrics: Dict, step: int):
        if not self.enabled:
            return

        # print("Logging metrics: ", metrics, mlflow.active_run().info.run_id)
        mlflow.log_metrics(metrics, step=step)
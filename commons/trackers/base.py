from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import mlflow
from mlflow.entities import RunStatus
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Generic
from commons.pipeline.types import Model


trackers_registry = {}

class Tracker(BaseModel, ABC):
    name: str
    enabled: bool = True
    run_id: Optional[str] = None

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, 'name', None)
        if name is None:
            raise ValueError(f"Default value of 'name' shall be set for tracker config class:{cls}")

        trackers_registry[name] = cls

    @abstractmethod
    def start_run(self, trackers: Model):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def end_run(self, error: bool = False):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def log_params_flatten(self, parent_key: str, params: Dict[str, Any]):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def log_artifacts(self, local_directory: str):
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict, step: int):
        raise NotImplementedError("Subclasses must implement this method")

    def watch(self, model, log_graph: bool = True):
        pass

    class Config:
        extra = "allow"


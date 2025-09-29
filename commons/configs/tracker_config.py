from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from commons.trackers.base import Tracker, trackers_registry


logging.basicConfig(
    format='%(asctime)s, %(levelname)-1s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:::%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingTrackersConfig(BaseModel):
    project_name: str
    experiment_name: str
    run_name: str
    run_description: str
    trackers: List[Tracker] = []

    def __init__(self, **kwargs):
        if kwargs.get('trackers') is not None:
            # instantiate specific TrackerConfig class
            for index in range(len(kwargs['trackers'])):
                current_tracker = kwargs['trackers'][index]
                if isinstance(current_tracker, dict):
                    target_name = current_tracker['name']
                    for name, subclass in trackers_registry.items():
                        if target_name == name:
                            current_tracker = subclass(**current_tracker)
                            break
                    kwargs['trackers'][index] = current_tracker

        super().__init__(**kwargs)

    def start_run(self):
        for tracker in self.trackers:
            try:
                print("Tracker: ", tracker)
                tracker.start_run(self)
            except Exception as e:
                logger.error(f"Error in start_run of tracker#{tracker.name} = {e}")

    def end_run(self, error: bool = False):
        for tracker in self.trackers:
            try:
                tracker.end_run(error)
            except Exception as e:
                logger.error(f"Error in end_run of tracker#{tracker.name} = {e}")

    def log_params(self, params: Dict[str, Any]):
        for tracker in self.trackers:
            try:
                tracker.log_params(params)
            except Exception as e:
                logger.error(f"Error in log_params of tracker#{tracker.name} = {e}")

    def log_params_flatten(self, parent_key: str, params: Dict[str, Any]):
        for tracker in self.trackers:
            try:
                tracker.log_params_flatten(parent_key, params)
            except Exception as e:
                logger.error(f"Error in log_params of tracker#{tracker.name} = {e}")

    def log_artifacts(self, local_directory: str):
        for tracker in self.trackers:
            try:
                tracker.log_artifacts(local_directory)
            except Exception as e:
                logger.error(f"Error in log_artifacts of tracker#{tracker.name} = {e}")

    def log_metrics(self, metrics: Dict, step: int):
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Error in log_metrics of tracker#{tracker.name} = {e}")

    def watch(self, model, log_graph: bool = True):
        for tracker in self.trackers:
            try:
                tracker.watch(model, log_graph)
            except Exception as e:
                logger.error(f"Error in watch of tracker#{tracker.name} = {e}")

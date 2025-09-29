import random
import string
import time
from datetime import datetime, timedelta

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
import enum
from typing import Optional, List

from hydra.core.plugins import Plugins

from commons.configs import model_config, training_strategy_config
from commons.configs.data_loader_config import DataLoaderConfig
from commons.hydra.hydra_plugins import HydraConfigsSearchPathPlugin
from commons.configs.model_config import ModelConfig
from commons.configs.trainer_config import ModelEvalConfig, ModelExportConfig, ModelInferenceConfig, ModelTrainConfig, TrainDatasetConfig
from commons.configs.training_strategy_config import TrainingStrategyConfig
from commons.configs.tracker_config import TrainingTrackersConfig

class TrainerPipelineConfig(BaseModel):
    model_version: str = Field(default_factory=lambda: str(int(time.time())))
    run_id: str = Field(default_factory=lambda: f"run_${datetime.now().strftime('%Y%m%d_%H%M%S')}_${''.join(random.choice(string.ascii_letters) for _ in range(4))}")
    log_verbosity: Optional[int] = 2
    model: ModelConfig
    training_strategy: TrainingStrategyConfig
    dataset: TrainDatasetConfig
    data_loader: DataLoaderConfig
    train: ModelTrainConfig
    inference: ModelInferenceConfig
    eval: ModelEvalConfig
    export: ModelExportConfig
    trackers: TrainingTrackersConfig
    config_str: str = None

    def __init__(self, **kwargs):
        # instantiate specific ModelConfig class
        current_model_impl = kwargs['model']
        if isinstance(current_model_impl, dict):
            model_kind = current_model_impl['kind']
            model_name = current_model_impl['name']
            target_name = f"{model_kind}/{model_name}"
            for name, subclass in model_config.model_registry.items():
                if target_name == name:
                    current_model_impl = subclass(**current_model_impl)
                    break
            kwargs['model'] = current_model_impl

        # instantiate specific TrainingStrategyConfig class
        current_training_strategy_impl = kwargs['training_strategy']
        if isinstance(current_training_strategy_impl, dict):
            target_name = current_training_strategy_impl['name']
            for name, subclass in training_strategy_config.training_strategy_registry.items():
                if target_name == name:
                    current_training_strategy_impl = subclass(**current_training_strategy_impl)
                    break
            kwargs['training_strategy'] = current_training_strategy_impl

        super().__init__(**kwargs)


def init_hydra():
    OmegaConf.register_new_resolver(
        "eval", lambda x: eval(x)
    )

    OmegaConf.register_new_resolver(
        "random_chars", lambda x: ''.join(random.choice(string.ascii_letters) for _ in range(x))
    )

    OmegaConf.register_new_resolver(
        "current_time", lambda: datetime.now()
    )

    # OmegaConf.register_new_resolver(
    #     "today", lambda x: datetime.today().strftime('%Y%m%d')
    # )

    OmegaConf.register_new_resolver(
        "day_before_days", lambda x: (datetime.today() - timedelta(days=x)).strftime('%Y%m%d')
    )

    Plugins.instance().register(HydraConfigsSearchPathPlugin)

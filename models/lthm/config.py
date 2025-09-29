import os.path
from typing import Optional, List, Tuple, Union, Mapping

from enum import Enum

import numpy as np
import pandas as pd
import functools
from pydantic import BaseModel
import pickle

from commons.pipeline.types import Stats
from commons.configs.model_config import ModelConfig, ModelKind
from commons.transformers.configs import TransformerConfig


class ProductTowerConfig(BaseModel):
    seq_emb_dim: int
    product_emb_dim: int
    

class LTHMModelConfig(ModelConfig):
    kind: ModelKind = ModelKind.CROSSDOMAIN
    # supported values: transformer_encoder
    type: str = "lthm_seq"
    name: str = "lthm"
    n_labels: int = 5
    n_pcc_buckets: int = 15
    max_pcc: float = 2.0
    n_watch_duration_buckets: int = 20
    max_watch_duration: float = 60.0
    lookahead: List[int] = [0, 20, 40, 60, 80, 100]
    detach_input_for_loss_calc: bool = False
    softmax_temperature: float = 1.0
    transformer_config: TransformerConfig
    metrics_k_all: List[int] = [1, 5, 20, 50]
    context_width: int = 150
    num_layers: int = 48
    use_snr_optim: bool = True
    lr: float = 6e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    train_mini_batch_size: int
    min_history_size: int = 1
    product_tower: ProductTowerConfig
    use_only_updated_data: bool
    knn_eval: bool = False

    @property
    def emb_dim(self) -> int:
        return self.transformer_config.attn_config.n_embd

    @property
    def export_tokens(self) -> int:
        return len(self.lookahead)

    @property
    def export_span(self) -> int:
        return max(self.lookahead) + 1

    def get_builder(self, stats: Optional[Stats]):
        from models.lthm.builder import LTHMModelBuilder
        return LTHMModelBuilder(stats, self)
    
    def custom_data_preprocessor(self, df: pd.DataFrame, kind: str = "train") -> pd.DataFrame:
        return df
    
    def special_data_prepreprocessor(self, df: pd.DataFrame, kind: str = "train") -> pd.DataFrame:
        return df
    
    def preprocess_fn(self, kind: str = "train"):
        def _preprocess_fn(df: pd.DataFrame):
            df = self.special_data_prepreprocessor(df, kind)
            df = self.features.default_data_mapper(df)
            df = self.custom_data_preprocessor(df, kind)
            return df
        return _preprocess_fn

    
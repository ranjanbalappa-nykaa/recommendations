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


class RankerModelConfig(ModelConfig):
    kind: ModelKind = ModelKind.RANKER
    type: str = "factorized_dlrm"
    name: str = "ranker_model"
    query_features: Optional[List[str]] = None
    item_features: Optional[List[str]] = None


    @property
    def product_features_list(self):
        if self.item_features is not None:
            return self.item_features
        else:
            all_features = self.features.categorical_features + self.features.numerical_features + \
                            self.features.bool_features + self.features.timestamp_features + \
                            self.features.one_hot_string_features
            
            return [f.name for f in all_features if f.tower_name.value == "product"]
        
    @property
    def query_features_list(self):
        if self.item_features is not None:
            return self.item_features
        else:
            all_features = self.features.categorical_features + self.features.numerical_features + \
                            self.features.bool_features + self.features.timestamp_features + \
                            self.features.one_hot_string_features
            
            return [f.name for f in all_features if f.tower_name.value == "query"]
        
    
    @property
    def user_features_list(self):
        if self.item_features is not None:
            return self.item_features
        else:
            all_features = self.features.categorical_features + self.features.numerical_features + \
                            self.features.bool_features + self.features.timestamp_features + \
                            self.features.one_hot_string_features
            
            return [f.name for f in all_features if f.tower_name.value == "user"]
        
    
    def get_builder(self, stats: Optional[Stats]):
        from ranker.builder import RankerModelBuilder
        return RankerModelBuilder(self, stats)
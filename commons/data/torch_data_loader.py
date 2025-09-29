import logging
from typing import Union, Any, Dict, Iterator, List, Optional

import pandas as pd
import numpy as np
import torch
import random
from torch.utils import data

from commons.configs.model_config import ModelConfig
from commons.configs.feature_config import FeaturesConfig
from commons.data.simple_dataset_generator import SimpleDatasetGenerator


def _coerce_to_shape(feature_name: str, value: List[np.ndarray], shape: tuple[int]) -> np.ndarray:
    sentinel = np.zeros(shape[1:])
    max_history_size = shape[0]
    for v in value:
        if isinstance(v, np.ndarray):
            assert v.shape == shape[1:], f'{feature_name} shapes do not match: {v.shape} != {shape[1:]}'
    if max_history_size > len(value):
        return np.stack(value + (max_history_size - len(value)) * [sentinel], axis=0)
    elif max_history_size < len(value):
        # FIXME This should not happen but often len(value) = max_history_size + 1 !!
        return np.stack(value[:max_history_size], axis=0)
    return np.stack(value, axis=0)


def _make_features_compliant(batch: Dict[str, Any], features_config: FeaturesConfig, is_series) -> None:
    """
    :param batch: Dict[str, Union[pd.Series, np.array, List[str]]
    :param is_series: boolean to indicate if all dict entries are wrapped in pd.Series
    :return: None (inplace ops)
    """
    for key in batch:
        if is_series:
            batch[key] = batch[key].values
            
        tensor_feature = features_config.get_tensor_feature(key)
        if tensor_feature is not None:
            result = []
            shape = None
            for value in batch[key]:
                value_ = np.array(value)
                if value_.shape == tensor_feature.get_emb_dim_as_shape():
                    result.append(value_)
                else:
                    result.append(_coerce_to_shape(key, value_.tolist(), tensor_feature.get_emb_dim_as_shape()))
                if shape is None:
                    shape = result[-1].shape
                assert shape == result[-1].shape, f"{key}'s shapes don't match: {shape} != {result[-1].shape}"
            batch[key] = np.stack(result, axis=0)

        tensor_list_feature = features_config.get_tensor_list_feature(key)
        if tensor_list_feature is not None:
            result = []
            shape = None
            for value in batch[key]:
                value_ = np.array(value)
                if value_.shape != tensor_list_feature.get_shape():
                    raise ValueError(f"shapes don't match! {value_.shape} != {tensor_list_feature.get_shape()}!")
                result.append(value_)
                if shape is None:
                    shape = result[-1].shape
                assert shape == result[-1].shape, f"{key}'s shapes don't match: {shape} != {result[-1].shape}"
            batch[key] = np.stack(result, axis=0)
        
        # Make one hot string feature compliant.
        one_hot_string_feature = features_config.get_one_hot_string_feature(key)
        if one_hot_string_feature is not None:
            result = []
            for value in batch[key]:
                value_ = np.array(value, dtype=np.int64)
                result.append(value_)
            batch[key] = np.stack(result, axis=0)


class GroupedDataframeWrapperDataset(data.IterableDataset):
    def __init__(
            self, 
            dataframe_generator: Iterator[pd.DataFrame],
            limit: int,
            batch_size: int,
            model_config: ModelConfig 
    ):
        super().__init__()
        self._custom_data_generator = dataframe_generator
        self._is_simple_generator = isinstance(dataframe_generator, SimpleDatasetGenerator)
        self._is_worker_info_set = False
        self._count = limit
        self._batch_size = batch_size
        self._model_config = model_config
    
    def set_torch_worker_info(self, worker_info):
        if isinstance(self._custom_data_generator, SimpleDatasetGenerator):
            self._custom_data_generator.set_torch_worker_info(worker_info)

   

    def __iter__(self):
        if self._is_simple_generator and not self._is_worker_info_set:
            worker_info = data.get_worker_info()
            if worker_info is not None:
                self.set_torch_worker_info(worker_info)
            self._is_worker_info_set = True

        # Given a dataframe convert into dict of feature names to tensors.
        def convert_to_tensor(batch: pd.DataFrame):
            batch = dict(batch)
            # Dict[str, Series] -> Dict[str, Union[np.array, List[str]]
            _make_features_compliant(batch, features_config=self._model_config.features, is_series=True)
            # We might have some string features which we need for retrieval inference.
            return {
                f: torch.from_numpy(batch[f])
                for f in batch
            }
        
        #iterator using generator
        group_config = self._group_config
        for df in self._custom_data_generator:
                gdf = df.groupby(by=group_config.group_by_columns)
                for _, rows in gdf:
                    rows.reset_index(inplace=True)
                    small_df = pd.DataFrame(rows)
                    if small_df.shape[0] < group_config.minimum_group_size:
                        continue
                    if group_config.maximum_group_size is not None and group_config.maximum_group_size < \
                            small_df.shape[0]:
                        continue
                    if group_config.sort_by_columns is not None and len(group_config.sort_by_columns) > 0:
                        small_df.sort_values(
                            by=group_config.sort_by_columns,
                            ascending=not group_config.sort_reverse,
                            inplace=True,
                        )
                    yield convert_to_tensor(small_df)
                    self._count -= 1
                    if self._count < 1:
                        break
                if self._count < 1:
                    break





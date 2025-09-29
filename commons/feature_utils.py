
from typing import Tuple
import numpy as np
import pandas as pd
import xxhash

MAX_LONG_VALUE_PLUS_ONE = 2 ** 63
CATEGORICAL_VAR_HASH_PAD_TOKEN = 0
NA_NUMERICAL_VALUE = -1.0
ONE_HOT_STRING_SIZE = 470 # TODO (Chetan) move this to OneHotStringFeature
ONE_HOT_STRING_ONES_MAX_LENGTH = 100 
ONE_HOT_STRING_ONES_PAD_TOKEN = -1 
ONE_HOT_POSITIVE_VALUE = "1"
ONE_HOT_STRING_DEFAULT = "0" * ONE_HOT_STRING_SIZE


def to_lower_case(name):
    return name.lower()


def pad_array(arr, size, pad_token=CATEGORICAL_VAR_HASH_PAD_TOKEN):
    arr = np.array(arr, dtype='long').reshape(-1,)
    arr = arr[:size]
    t = max(0, size - len(arr))
    return np.pad(arr, pad_width=(0, t), mode='constant', constant_values=pad_token)


def get_id_score_list_feature_key(feature_name: str):
    return f'{feature_name}_keys'


def get_id_score_list_feature_value(feature_name: str):
    return f'{feature_name}_values'


def hash_feature_name_to_int(feature_name: str):
    return xxhash.xxh32(to_lower_case(feature_name), 0).intdigest()


def hash_string_to_long(arg: str, seed: int, value_to_lower: bool):
    # xxh64 ranges from 0 to 2**64-1
    # torch.int64 ranges from -2**63 to 2**63-1
    arg = str(arg)
    if value_to_lower:
        arg = arg.lower()
    return xxhash.xxh64(arg, seed).intdigest() - MAX_LONG_VALUE_PLUS_ONE


def fix_na_bool(batch: pd.DataFrame, column: str):
    batch[column] = batch[column].values.astype(np.float32)


def fix_na_str(batch: pd.DataFrame, column: str):
    # note we will lower case NA to na for 'string_lower' dtype before hashing
    # See https://gitlab.dailyhunt.in/josh-p13n/feed-engine-ml-infer-py/blob/
    # master-multi-candidates-pv3-refactor-poetry-mtl-refactor-json/recsys/pre_ranker/col_value_parser.py
    batch[column].fillna("NA", inplace=True)


def fix_na_int64_lower(batch: pd.DataFrame, column: str):
    seed = hash_feature_name_to_int(column)
    na_value = hash_string_to_long("NA", seed, value_to_lower=True)
    batch[column] = batch[column].apply(lambda x: na_value if x is None else x).astype(np.int64)


def fix_na_int64_upper(batch: pd.DataFrame, column: str):
    seed = hash_feature_name_to_int(column)
    na_value = hash_string_to_long("NA", seed, value_to_lower=False)
    batch[column] = batch[column].apply(lambda x: na_value if x is None else x).astype(np.int64)


def fix_na_string_list(batch: pd.DataFrame, column: str):
    batch[column] = batch[column].apply(lambda x: [] if x is None else x)


def fix_na_one_hot_string(batch: pd.DataFrame, column: str):
    batch[column] = batch[column].apply(lambda x: ONE_HOT_STRING_DEFAULT if x is None else x)


def fix_na_tensor(batch: pd.DataFrame, column: str, emb_dim: int):
    sentinel_vector = np.zeros(emb_dim)
    batch[column] = batch[column].apply(lambda x: sentinel_vector if x is None else x)


def fix_na_tensor_list(batch: pd.DataFrame, column: str, shape: Tuple[int, ...]):
    sentinel_vector = np.zeros((int(np.prod(shape)),), dtype=np.float32)
    batch[column] = batch[column].apply(lambda x: sentinel_vector if x is None else
                                        np.array(x[0] if hasattr(x[0], '__len__') else x, dtype=np.float32))


def fix_partial_tensor_list(batch: pd.DataFrame, column: str, shape: Tuple[int, ...]):
    numel = int(np.prod(shape))

    def _func(x: np.ndarray):
        if int(np.prod(x.shape)) == numel:
            return x.reshape(shape)
        x = x.reshape(-1, *shape[1:])
        if shape[0] < x.shape[0]:
            return x[:shape[0]]
        residual_shape = (shape[0] - x.shape[0], *shape[1:])
        return np.concatenate((x, np.zeros(residual_shape, dtype=np.float32)), axis=0)
    batch[column] = batch[column].apply(_func)


def fill_na(batch: pd.DataFrame):
    batch.fillna(NA_NUMERICAL_VALUE, inplace=True)


def rename_column(batch: pd.DataFrame, src_column: str, target_column: str):
    batch.rename({src_column: target_column}, inplace=True, axis=1)


def copy_value(batch: pd.DataFrame, src_column: str, target_column: str):
    batch[target_column] = batch[src_column]  # .copy()

# Get indices where the string contains "1"
def create_array_one_hot_feature(batch: pd.DataFrame, column: str):    
    arr = []
    for val in batch[column].values.tolist():
        indices = [i  for i in range(len(val)) if  val[i] == ONE_HOT_POSITIVE_VALUE]
        row = pad_array(arr=indices[:ONE_HOT_STRING_ONES_MAX_LENGTH], size=ONE_HOT_STRING_ONES_MAX_LENGTH, pad_token=ONE_HOT_STRING_ONES_PAD_TOKEN)
        arr.append(row)
    batch[column] = arr


def box_lat_long_feature(batch: pd.DataFrame, column: str):
    arr = []
    for val in batch[column].values.tolist():
        try:
            float_val = float(val)
        except:
            float_val = -1
        arr.append(float_val)
    batch[column] = np.array(arr)


def transform_value_to_lower(batch: pd.DataFrame, column: str):
    batch[column] = batch[column].apply(str.lower)


def xxhash_categorical_values_to_number(batch: pd.DataFrame, column: str, value_to_lower: bool):
    seed = hash_feature_name_to_int(feature_name=column)
    batch[column] = np.array([
        hash_string_to_long(arg=value, seed=seed, value_to_lower=value_to_lower)
        for value in batch[column].values
    ], dtype=np.int64)


def handle_categorical_history_feature(
        batch: pd.DataFrame,
        column: str,
        hash_ids: bool,
        history_length: int,
        history_id_feature_name: str,
        remove_history_id_from_history: bool = False,
):
    if not hash_ids and not remove_history_id_from_history:
        return truncate_and_pad_to_fix_len(batch=batch, column=column, length=history_length)

    seed = hash_feature_name_to_int(feature_name=history_id_feature_name)
    processed = []
    # 1. Optionally Hash all the history ids
    # 2. Remove the history_id_feature_name feature from history. Here is it assumed that the history_id_feature_name is processed (hashed) already.
    # 3. Cap history to history_length
    # 4. Pad history to history_length
    for history_id_feature, history in zip(batch[history_id_feature_name].values, batch[column].values):
        found = 0
        row_history = []
        for h in history:
            if hash_ids:
                h = hash_string_to_long(arg=h, seed=seed, value_to_lower=False)
            if remove_history_id_from_history and h == history_id_feature:
                continue
            row_history.append(h)
            found += 1
            if found == history_length:
                break
        processed.append(pad_array(arr=row_history, size=history_length))
    batch[column] = processed


def truncate_and_pad_to_fix_len(batch: pd.DataFrame, column: str, length: int):
    batch[column] = batch[column].apply(lambda x: pad_array(arr=x, size=length))

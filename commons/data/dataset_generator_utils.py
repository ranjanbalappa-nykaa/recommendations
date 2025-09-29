import glob
import math
import numpy as np
import pandas as pd
from typing import List, Optional
import socket

from commons.data.data_store import DataStoreAccessor, get_date_range_str
from commons.configs.trainer_config import TrainDatasetConfig, FileSystemConfig
from commons.pipeline.types import DfMapperFn


def get_paths_for_worker(
        worker_id: int,
        data_paths: List[str],
        num_workers: int,
        seed: Optional[int] = None,
) -> List[str]:
    data_paths = list(sorted(data_paths))
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(data_paths)
    total_paths = len(data_paths)
    paths_per_worker = math.floor(total_paths / num_workers)
    rem_paths = total_paths % num_workers
    paths_cur_worker = paths_per_worker + (1 if rem_paths > worker_id else 0)
    start_range = worker_id * paths_per_worker + min(rem_paths, worker_id)
    end_range = min(total_paths, start_range + paths_cur_worker)
    paths = data_paths[start_range:end_range]

    print(
        f"Worker#{worker_id} - start_range:{start_range} end_range:{end_range} paths:{len(paths)}")

    return paths


def get_path_chunks(
        paths: List[str],
        block_size: int,
        shuffle_files: bool = False
):
    paths = np.array(paths)
    if shuffle_files:
        np.random.shuffle(paths)

    num_segments = len(paths) // block_size
    if num_segments == 0:
        num_segments = 1
    return [p.tolist() for p in np.array_split(paths, num_segments)]


def read_table_impl_one_path(path: str, columns: List[str], data_mapper: DfMapperFn, fs_config: FileSystemConfig) -> pd.DataFrame:
    data_store = DataStoreAccessor.get_instance(fs_config)
    df = data_store.read_single_parquet_file(path, columns=columns)
    return data_mapper(df)

def strip_path(path: str):
    if path.startswith('s3://'):
        return path[5:]
    return path




def get_train_data_paths(dataset_config: TrainDatasetConfig):
    if dataset_config.path_glob_train != "":
        return list(glob.glob(dataset_config.path_glob_train))
    data_dates = get_date_range_str(date=dataset_config.train_data_end_date,
                                    steps=dataset_config.train_period_in_days,
                                    backward=True)
    if len(dataset_config.exclude_dates) > 0:
        print(f"Excluding dates from training {dataset_config.exclude_dates}")
        data_dates = [data_date for data_date in data_dates if data_date not in dataset_config.exclude_dates]
        print(f"Only including the following train dates: {data_dates}")
    assert len(data_dates) > 0, "Train data ds range is empty!"
    data_store = DataStoreAccessor.get_instance(dataset_config.filesystem_config)
    data_paths = data_store.get_training_data_paths_for_dates(data_dates=data_dates, data_ratio=dataset_config.train_data_ratio)
    print(f"Got total overall paths for kind: training", socket.gethostname(), len(data_paths))
    return data_paths


def get_val_data_paths(dataset_config: TrainDatasetConfig, for_extra_day=False) -> List[str]:
    if dataset_config.path_glob_test != "":
        return list(glob.glob(dataset_config.path_glob_test))
    if not for_extra_day:
        data_dates = get_date_range_str(date=dataset_config.val_data_start_date,
                                        steps=dataset_config.val_period_in_days,
                                        backward=False)
    else:
        if dataset_config.extra_day_val_data_start_date is not None and dataset_config.extra_day_val_period_in_days > 0:
            data_dates = get_date_range_str(date=dataset_config.extra_day_val_data_start_date,
                                            steps=dataset_config.extra_day_val_period_in_days,
                                            backward=False)
        else:
            return []
    if len(dataset_config.exclude_dates) > 0:
        print(f"Excluding dates from validation {dataset_config.exclude_dates}")
        data_dates = [data_date for data_date in data_dates if data_date not in dataset_config.exclude_dates]
        print(f"Only including the following val dates: {data_dates}")
    assert len(data_dates) > 0, "Eval data ds range is empty!"
    data_store = DataStoreAccessor.get_instance(dataset_config.filesystem_config)
    data_paths = data_store.get_training_data_paths_for_dates(data_dates=data_dates, data_ratio=dataset_config.val_data_ratio)
    print(f"Got total overall paths for kind: validation", socket.gethostname(), len(data_paths))
    return data_paths
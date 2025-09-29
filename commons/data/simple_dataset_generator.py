from typing import List
import pandas as pd
import functools

from commons.data.dataset_generator_utils import (
    get_path_chunks,
    read_table_impl_one_path
)
from commons.pipeline.types import DfMapperFnForKind
from commons.configs.trainer_config import FileSystemConfig


class SimpleDatasetGenerator:
    def __init__(
            self,
            kind: str,
            node_id: str,
            worker_id: int,
            paths: List[str],
            block_size: int,
            max_readers: int,
            columns: List[str],
            data_mapper: DfMapperFnForKind,
            fs_config: FileSystemConfig,
            shuffle_files: bool = True,
            shuffle_data: bool = False,
    ):
        self.kind = kind
        self.node_id = node_id
        self.worker_id = worker_id
        self.block_size = block_size
        assert max_readers == 1
        self.max_readers = max_readers
        self.columns = columns
        self.data_mapper = data_mapper
        self.fs_config = fs_config
        self.path_chunks = get_path_chunks(paths, block_size, shuffle_files)
        self.shuffle_data = shuffle_data

    def set_torch_worker_info(self, worker_info):
        if self.torch_worker_id is None:
            self.torch_worker_id = worker_info.id
            self.torch_num_workers = worker_info.num_workers
            _ = self.filtered_path_chunks

    @functools.cached_property
    def filtered_path_chunks(self):
        if self.torch_worker_id is None:
            return self.path_chunks
        return [p for i, p in enumerate(self.path_chunks) if i % self.torch_num_workers == self.torch_worker_id]

    def __iter__(self):
        df_mapper_fn = self.data_mapper(self.kind)
        for chunk in self.filtered_path_chunks:
            dfs = list(filter(lambda x: x is not None, [
                read_table_impl_one_path(
                    path=f, 
                    columns=self.columns, 
                    data_mapper=df_mapper_fn, fs_config=self.fs_config) \
                for f in chunk
            ]))

            if len(dfs) > 0:
                df = pd.concat(dfs, axis=0)
                if self.shuffle_data:
                    yield df.sample(frac=1.0)
                else:
                    yield df
        
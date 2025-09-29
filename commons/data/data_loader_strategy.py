from typing import Generic, List
import ray
from abc import abstractmethod
from overrides import override

from commons.configs.data_loader_config import DataLoaderConfig
from commons.configs.trainer_config import FileSystemConfig
from commons.pipeline.types import DataGenerator, DfMapperFnForKind
from commons.data.simple_dataset_generator import SimpleDatasetGenerator


class DataLoaderStrategy(Generic[DataGenerator]):
    def __init__(
            self,
            data_loader_config: DataLoaderConfig,
            columns: List[str],
            data_mapper: DfMapperFnForKind
    ):
        self.columns = columns
        self.data_loader_config = data_loader_config
        self.data_mapper = data_mapper

    @abstractmethod
    def load(self, kind: str, worker_id: int, paths: List[str], fs_config: FileSystemConfig) -> DataGenerator:
        raise NotImplementedError("Subclasses must implement this method")
    

class SimpleDataLoaderStrategy(DataLoaderStrategy[SimpleDatasetGenerator]):
    def __init__(
            self,
            data_loader_config: DataLoaderConfig,
            columns: List[str],
            data_mapper: DfMapperFnForKind,
    ):
        super().__init__(data_loader_config, columns, data_mapper)

    @override
    def load(self, kind: str, worker_id: int, paths: List[str], fs_config: FileSystemConfig):
        generator = SimpleDatasetGenerator(
            kind=kind,
            node_id=ray.get_runtime_context().node_id,
            worker_id=worker_id,
            paths=paths,
            block_size=self.data_loader_config.block_size,
            # parallelism needs to orchestrated outside
            max_readers=1,
            columns=self.columns,
            data_mapper=self.data_mapper,
            fs_config=fs_config,
            shuffle_files=self.data_loader_config.shuffle_files,
            shuffle_data=self.data_loader_config.shuffle_data
        )
        return generator 
from typing import List, Union
from torch.utils import data


from commons.configs.data_loader_config import DataLoaderConfig, DataLoaderKind
from commons.configs.model_config import ModelConfig
from commons.configs.trainer_config import FileSystemConfig
from commons.data.data_loader_strategy import (
    DataLoaderStrategy,
    SimpleDataLoaderStrategy,
)
from commons.data.torch_data_loader import GroupedDataframeWrapperDataset
from commons.pipeline.types import DfMapperFnForKind


def get_data_loader_strategy(
        data_loader_config: DataLoaderConfig,
        columns: List[str],
        data_mapper: DfMapperFnForKind
) -> DataLoaderStrategy:
    return SimpleDataLoaderStrategy(data_loader_config, columns, data_mapper)


def get_torch_dataloader(
        kind: str,
        worker_id: int,
        paths: List[str],
        batch_size: int,
        num_steps: int,
        data_loader_strategy: DataLoaderStrategy,
        model_config: ModelConfig,
        fs_config: FileSystemConfig,
) -> Union[data.DataLoader, data.Dataset]:
    assert kind in ['val', 'train'], "kind must be train or val"
    generator = data_loader_strategy.load(kind, worker_id, paths, fs_config)
    
    print("Returning grouped DS generator")
    if kind != 'val' and data_loader_strategy.data_loader_config.max_readers > 0:
        parallelism = data_loader_strategy.data_loader_config.max_readers
        # These parameters are only supported if torch multiprocess data loading is used.
        kwargs = {
            'num_workers': parallelism,
            'prefetch_factor': data_loader_strategy.data_loader_config.max_prefetch,
        }
    else:
        kwargs = {}

    if data_loader_strategy.data_loader_config.bypass_dataloader:
        return GroupedDataframeWrapperDataset(
            dataframe_generator=generator,
            limit=num_steps,
            batch_size=batch_size,
            model_config=model_config
        )
    
    return data.DataLoader(
        GroupedDataframeWrapperDataset(
            dataframe_generator=generator,
            limit=num_steps,
            batch_size=batch_size,
            model_config=model_config),
        batch_size=None,
        pin_memory=True,
        **kwargs,
    )

    
    
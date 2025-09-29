import enum

from pydantic import BaseModel


class DataLoaderKind(str, enum.Enum):
    SIMPLE = "simple"


class DataLoaderConfig(BaseModel):
    kind: DataLoaderKind
    block_size: int
    max_prefetch: int
    max_readers: int
    shuffle_files: bool
    shuffle_data: bool
    # only used for simple dataloader
    mini_batch_size: int
    shuffle_buffer_num_mini_batches: int
    macro_batches_multiples: int
    pin_memory: bool = True
    bypass_dataloader: bool = False
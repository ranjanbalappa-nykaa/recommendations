from collections import namedtuple
from typing import Callable, Dict, List, TypeVar, Union

import pandas as pd

NormalizationStats = namedtuple('NormalizationStats', ['mean', 'std_dev'])
Stats = namedtuple('Stats', ['quantile_stats', 'normalization_stats', 'tensor_stats', 'batch'])

Model = TypeVar("Model")
DataGenerator = TypeVar("DataGenerator")

DfMapperFn = Callable[[pd.DataFrame], pd.DataFrame]
DfMapperFnForKind = Callable[[str], DfMapperFn]
QuantileStats = Dict[str, List[float]]
Shape = Union[int, list[int]]
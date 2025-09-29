
import enum
import functools
from typing import Any, Callable, ClassVar, List, Dict, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from commons import feature_utils


class EmbeddingTable(BaseModel):
    # table_name: str
    num_embeddings: int
    emb_dim: int
    use_qr: bool


class NumericalFeaturesDefaults(BaseModel):
    embed_feature: Optional[bool] = None


class CategoricalValueToNumberMapper(BaseModel):
    kind: str
    # we can also implement lookup

    registry: ClassVar[dict] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__fields__['kind'].default
        if kind is None:
            raise ValueError(f"Default value of 'kind' shall be set for CategoricalValueToNumberMapper sub class:{cls}")

        cls.registry[kind] = cls


# class XXHashSize(enum.Enum):
#     XXH32 = 32
#     XXH64 = 64
#     XXH128 = 128


class XXHashMapper(CategoricalValueToNumberMapper):
    kind: str = "xxhash"
    # size: XXHashSize = XXHashSize.XXH64


class NoneMapper(CategoricalValueToNumberMapper):
    kind: str = "none"


class CategoricalFeaturesDefaults(BaseModel):
    embedding: Optional[EmbeddingTable] = None
    proj_dim: Optional[int] = None
    value_to_number_mapper: Optional[CategoricalValueToNumberMapper] = None
    default_dtype: Optional[str] = None
    transform_value_to_lowercase: Optional[bool] = True

    def __init__(self, **kwargs):
        # instantiate specific CategoricalValueToNumberMapper class
        field = 'value_to_number_mapper'
        current_source_impl = kwargs.get(field)
        if current_source_impl is not None and isinstance(current_source_impl, dict):
            target_name = current_source_impl['kind']
            for name, subclass in CategoricalValueToNumberMapper.registry.items():
                if target_name == name:
                    current_source_impl = subclass(**current_source_impl)
                    break
            kwargs[field] = current_source_impl

        super().__init__(**kwargs)


class CategoricalHistoryFeatureDefaults(CategoricalFeaturesDefaults):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TensorFeaturesDefaults(BaseModel):
    emb_dim: Optional[int] = None


class TensorListFeaturesDefaults(BaseModel):
    shape: Optional[Tuple[int, ...]] = None


class BoolFeaturesDefaults(BaseModel):
    emb_dim: Optional[int] = None


class TimestampFeaturesDefaults(BaseModel):
    emb_dim: Optional[int] = None


class LatLongFeaturesDefaults(BaseModel):
    emb_dim: Optional[int] = None


class OneHotStringFeaturesDefaults(BaseModel):
    pass
    # TODO (Chetan): Add default length here.


class EmbeddingTableConfig(BaseModel):
    shared: Optional[Dict[str, EmbeddingTable]] = None
    query: Optional[Dict[str, EmbeddingTable]] = None
    item: Optional[Dict[str, EmbeddingTable]] = None

class FeatureDefaults(BaseModel):
    do_not_fix_na_values: bool = False
    transform_all_feature_names_to_lowercase: bool = True
    embedding_table_config: Optional[EmbeddingTableConfig] = None
    bool_features: Optional[BoolFeaturesDefaults] = None
    numerical_features: Optional[NumericalFeaturesDefaults] = None
    categorical_features: Optional[CategoricalFeaturesDefaults] = None
    categorical_history_features: Optional[CategoricalHistoryFeatureDefaults] = None
    tensor_features: Optional[TensorFeaturesDefaults] = None
    tensor_list_features: Optional[TensorListFeaturesDefaults] = None
    timestamp_features: Optional[TimestampFeaturesDefaults] = None
    lat_lng_features: Optional[LatLongFeaturesDefaults] = None
    one_hot_string_features: Optional[OneHotStringFeaturesDefaults] = None


class FeatureSourceKind(str, enum.Enum):
    INPUT = "input"
    DERIVED = "derived"


class FeatureTowerName(str, enum.Enum):
    QUERY = "query"
    PRODUCT = "product"
    CONTEXT = "context"
    OTHER = "other"


class FeatureSource(BaseModel):
    kind: FeatureSourceKind
    dtype: Optional[str] = None
    registry: ClassVar[dict] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__fields__['kind'].default
        if kind is None:
            raise ValueError(f"Default value of 'kind' shall be set for feature source sub class:{cls}")

        FeatureSource.registry[kind] = cls


class InputFeatureSource(FeatureSource):
    kind: FeatureSourceKind = FeatureSourceKind.INPUT
    input_field: Optional[str] = None


class DerivedFeatureSource(FeatureSource):
    kind: FeatureSourceKind = FeatureSourceKind.DERIVED
    # TODO: later add derived function


class FeatureKind(str, enum.Enum):
    Bool = "bool"
    Numerical = "numerical"
    Categorical = "categorical"
    CategoricalList = "categorical_list"
    CategoricalHistory = "categorical_history"
    Tensor = "tensor"
    TensorList = "tensor_list"
    Timestamp = "timestamp"
    LatLong = "latlong"
    OneHotString = "one_hot_string"


class Feature(BaseModel):
    name: str
    kind: FeatureKind
    source: FeatureSource = Field(default_factory=InputFeatureSource)
    do_not_convert_to_platform_type: bool = False
    include_in_eval_output: bool = False
    tower_name: FeatureTowerName = FeatureTowerName.OTHER

    registry: ClassVar[dict] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__fields__['kind'].default
        if kind is None and cls.__name__ != 'Task':
            print(f"Default value of 'kind' shall be set for feature sub class:{cls}")
            return

        Feature.registry[kind] = cls

    def __init__(self, **kwargs):
        # instantiate specific FeatureSource class
        current_source_impl = kwargs.get('source')
        if current_source_impl is not None and isinstance(current_source_impl, dict):
            target_name = current_source_impl['kind']
            for name, subclass in FeatureSource.registry.items():
                if target_name == name:
                    current_source_impl = subclass(**current_source_impl)
                    break
            kwargs['source'] = current_source_impl

        super().__init__(**kwargs)

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        if self.source.input_field is None:
            self.source.input_field = self.name

        if feature_defaults.transform_all_feature_names_to_lowercase \
                and any(ele.isupper() for ele in self.name) \
                and isinstance(self.source, InputFeatureSource):
            self.name = self.name.lower()


class Task(Feature):
    num_labels: int = 1
    weight: float = 1.0
    detached_estimator: bool = False


class BoolFeature(Feature):
    kind: FeatureKind = FeatureKind.Bool
    emb_dim: Optional[int] = None

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "bool"

        if feature_defaults.bool_features is None:
            return

        if feature_defaults.bool_features.emb_dim is None:
            return

        if self.emb_dim is None:
            self.emb_dim = feature_defaults.bool_features.emb_dim


class NumericalFeature(Feature):
    kind: FeatureKind = FeatureKind.Numerical
    embed_feature: Optional[bool] = None

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "float32"

        if feature_defaults.numerical_features is None:
            return

        if feature_defaults.numerical_features.embed_feature is None:
            return

        if self.embed_feature is None:
            self.embed_feature = feature_defaults.numerical_features.embed_feature


class OneHotStringFeature(Feature):
    kind: FeatureKind = FeatureKind.OneHotString
    # TODO (Chetan): Add feature length as param.

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        defaults = feature_defaults.one_hot_string_features
        assert (self.source.dtype == "one_hot_string") or (self.source.dtype is None), \
            "OneHotStringFeature is provided improper dtype"

        self.source.dtype = "one_hot_string"


class CategoricalFeature(Feature):
    kind: FeatureKind = FeatureKind.Categorical
    emb_table_name: Optional[str] = None
    proj_dim: Optional[int] = None
    transform_value_to_lowercase: Optional[bool] = None
    value_to_number_mapper: Optional[CategoricalValueToNumberMapper] = None

    def __init__(self, **kwargs):
        # instantiate specific CategoricalValueToNumberMapper class
        field = 'value_to_number_mapper'
        current_source_impl = kwargs.get(field)
        if current_source_impl is not None and isinstance(current_source_impl, dict):
            target_name = current_source_impl['kind']
            for name, subclass in CategoricalValueToNumberMapper.registry.items():
                if target_name == name:
                    current_source_impl = subclass(**current_source_impl)
                    break
            kwargs[field] = current_source_impl

        super().__init__(**kwargs)

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        defaults = feature_defaults.categorical_features

        if self.transform_value_to_lowercase is None:
            if defaults is not None and defaults.transform_value_to_lowercase is not None:
                self.transform_value_to_lowercase = defaults.transform_value_to_lowercase

        if self.source.dtype is None:
            if defaults is not None and defaults.default_dtype is not None:
                self.source.dtype = defaults.default_dtype
            else:
                self.source.dtype = "string_lower" if self.transform_value_to_lowercase else "string"

        if defaults is None:
            return

        if self.value_to_number_mapper is None and defaults.value_to_number_mapper is not None:
            self.value_to_number_mapper = defaults.value_to_number_mapper

        if self.proj_dim is None and defaults.proj_dim is not None:
            self.proj_dim = defaults.proj_dim

        if self.emb_table_name is None and defaults.embedding is not None:
            self.emb_table_name = "default_categorical"


class CategoricalHistoryFeature(Feature):
    """List of categorical values. Default dtype is string_list."""
    kind: str = FeatureKind.CategoricalHistory
    emb_table_name: Optional[str] = None
    history_length: int = 20
    history_id_feature_name: str
    value_to_number_mapper: Optional[CategoricalValueToNumberMapper] = None
    remove_history_id_from_history: bool = False

    def __init__(self, **kwargs):
        # instantiate specific CategoricalValueToNumberMapper class
        field = 'value_to_number_mapper'
        current_source_impl = kwargs.get(field)
        if current_source_impl is not None and isinstance(current_source_impl, dict):
            target_name = current_source_impl['kind']
            for name, subclass in CategoricalValueToNumberMapper.registry.items():
                if target_name == name:
                    current_source_impl = subclass(**current_source_impl)
                    break
            kwargs[field] = current_source_impl

        super().__init__(**kwargs)

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        defaults = feature_defaults.categorical_history_features
        if self.source.dtype is None:
            if defaults is not None and defaults.default_dtype is not None:
                self.source.dtype = defaults.default_dtype
            else:
                self.source.dtype = "string_list"

        if defaults is None:
            return

        if self.value_to_number_mapper is None and defaults.value_to_number_mapper is not None:
            self.value_to_number_mapper = defaults.value_to_number_mapper


class TensorFeature(Feature):
    kind: FeatureKind = FeatureKind.Tensor
    emb_dim: int = 0

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "tensor"

        if feature_defaults.tensor_features is None:
            return

        if feature_defaults.tensor_features.emb_dim is None:
            return

        if self.emb_dim == 0:
            self.emb_dim = feature_defaults.tensor_features.emb_dim

    def get_emb_dim_as_shape(self) -> tuple[int]:
        return self.emb_dim,


class TensorListFeature(Feature):
    kind: FeatureKind = FeatureKind.TensorList
    shape: Tuple[int, ...]

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "tensor_list"

        if feature_defaults.tensor_list_features is None:
            return

        if feature_defaults.tensor_list_features.shape is None:
            return

        if self.shape == tuple():
            self.shape = feature_defaults.tensor_list_features.shape

    def get_shape(self) -> Tuple[int, ...]:
        return self.shape


class TimestampFeature(Feature):
    kind: FeatureKind = FeatureKind.Timestamp
    emb_dim: Optional[int] = None

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "int64"

        if feature_defaults.timestamp_features is None:
            return

        if feature_defaults.timestamp_features.emb_dim is None:
            return

        if self.emb_dim is None:
            self.emb_dim = feature_defaults.timestamp_features.emb_dim


class LatLongFeature(Feature):
    kind: FeatureKind = FeatureKind.LatLong
    emb_dim: Optional[int] = None

    def populate_defaults(self, feature_defaults: FeatureDefaults):
        super().populate_defaults(feature_defaults)
        if self.source.dtype is None:
            self.source.dtype = "float32"

        if feature_defaults.lat_lng_features is None:
            return

        if feature_defaults.lat_lng_features.emb_dim is None:
            return

        if self.emb_dim is None:
            self.emb_dim = feature_defaults.lat_lng_features.emb_dim


class GroupDatasetConfig(BaseModel):
    group_by_columns: List[str] = Field(default_factory=list)
    sort_by_columns: List[str] = Field(default_factory=list)
    sort_reverse: bool = True
    flatten: bool = False
    minimum_group_size: int = 0
    maximum_group_size: Optional[int] = None


# TODO: support id_score_list features
class FeaturesConfig(BaseModel):
    defaults: FeatureDefaults
    embedding_table_config: EmbeddingTableConfig = Field(default_factory=EmbeddingTableConfig)
    embedding_tables: dict[str, EmbeddingTable] = Field(default_factory=dict)
    bool_features: list[BoolFeature] = Field(default_factory=list)
    numerical_features: list[NumericalFeature] = Field(default_factory=list)
    one_hot_string_features: list[OneHotStringFeature] = Field(default_factory=list)
    categorical_features: list[CategoricalFeature] = Field(default_factory=list)
    # will be deprecated in favor of CategoricalListFeature
    categorical_history_features: list[CategoricalHistoryFeature] = Field(default_factory=list)
    tensor_features: list[TensorFeature] = Field(default_factory=list)
    tensor_list_features: list[TensorListFeature] = Field(default_factory=list)
    timestamp_features: list[TimestampFeature] = Field(default_factory=list)
    lat_lng_features: list[LatLongFeature] = Field(default_factory=list)
    extra_eval_output_fields: list[Feature] = Field(default_factory=list)
    extra_input_fields: list[Feature] = Field(default_factory=list)
    group_dataset: Optional[GroupDatasetConfig] = None

    # these are calculated in init...
    input_columns: list[str] = list()  # Field(default_factory=list)
    input_to_feature_map: dict[str, list[Feature]] = dict()  # Field(default_factory=dict)
    features_map: dict[str, Feature] = dict()  # Field(default_factory=dict)
    dtypes: dict[str, str] = dict()  # Field(default_factory=dict)
    dtypes_string_map: dict[str, str] = dict()  # Field(default_factory=dict)
    transformers: list[Callable[[pd.DataFrame], None]] = list()  # Field(default_factory=list)

    def __init__(self, **kwargs):
        # instantiate specific Feature class
        for field in ['extra_eval_output_fields', 'extra_input_fields']:
            if kwargs.get(field) is not None:
                for index in range(len(kwargs[field])):
                    current_feature = kwargs[field][index]
                    if isinstance(current_feature, dict):
                        target_name = current_feature['kind']
                        for name, subclass in Feature.registry.items():
                            if target_name == name:
                                current_feature = subclass(**current_feature)
                                break
                        kwargs[field][index] = current_feature

        super().__init__(**kwargs)

        if self.defaults.categorical_features is not None and self.defaults.categorical_features.embedding is not None:
            self.embedding_tables["default_categorical"] = self.defaults.categorical_features.embedding
        
        self.embedding_table_config = self.defaults.embedding_table_config
        
        input_columns = set()
        for feature in sum([self.bool_features,
                            self.numerical_features,
                            self.categorical_features,
                            self.categorical_history_features,
                            self.tensor_features,
                            self.tensor_list_features,
                            self.timestamp_features,
                            self.lat_lng_features,
                            self.one_hot_string_features,
                            self.extra_eval_output_fields,
                            self.extra_input_fields], []):
            if not isinstance(feature.source, DerivedFeatureSource):
                feature.populate_defaults(self.defaults)

            # TODO: handle same feature name occurring multiple times
            if isinstance(feature.source, InputFeatureSource):
                input_field = feature.source.input_field
                features = self.input_to_feature_map.get(input_field)
                if features is None:
                    features = []
                    input_columns.add(input_field)
                else:
                    existing_dtype = self.dtypes[input_field]
                    if existing_dtype != feature.source.dtype:
                        raise ValueError(f"Input field ({input_field}) with 2 types found - {existing_dtype} and {feature.source.dtype}")

                features.append(feature)
                self.input_to_feature_map[input_field] = features
                self.dtypes[input_field] = feature.source.dtype
                self.features_map[feature.name] = feature
                if feature.source.dtype == 'string' or feature.source.dtype == 'string_lower':
                    self.dtypes_string_map[input_field] = 'string'

        self.input_columns = list(input_columns)

        # TODO: throw error is some required values are not set
        """Fix NA values for all types"""
        if not self.defaults.do_not_fix_na_values:
            # fix na values
            for column in self.input_columns:
                if self.dtypes[column] == 'bool':
                    self.transformers.append(functools.partial(feature_utils.fix_na_bool, column=column))
                elif self.dtypes[column] == 'string' or self.dtypes[column] == 'string_lower':
                    self.transformers.append(functools.partial(feature_utils.fix_na_str, column=column))
                elif self.dtypes[column] == 'tensor':
                    emb_dim = max([f.emb_dim for f in self.input_to_feature_map[column] if isinstance(f, TensorFeature)]) or 0
                    self.transformers.append(functools.partial(feature_utils.fix_na_tensor, column=column, emb_dim=emb_dim))
                elif self.dtypes[column] == 'tensor_list':
                    shapes = [f.shape for f in self.input_to_feature_map[column] if isinstance(f, TensorListFeature)]
                    if len(shapes) > 0:
                        shape = shapes[0]
                        self.transformers.append(functools.partial(feature_utils.fix_na_tensor_list,
                                                                   column=column,
                                                                   shape=shape))
                        self.transformers.append(functools.partial(feature_utils.fix_partial_tensor_list,
                                                                   column=column,
                                                                   shape=shape))
                elif self.dtypes[column] == 'string_list' or self.dtypes[column] == 'int64_list':
                    self.transformers.append(functools.partial(feature_utils.fix_na_string_list, column=column))
                elif self.dtypes[column] == 'int64':
                    self.transformers.append(functools.partial(feature_utils.fix_na_int64_lower, column=column))
                elif self.dtypes[column] == 'int64_upper':
                    self.transformers.append(functools.partial(feature_utils.fix_na_int64_upper, column=column))
                elif self.dtypes[column] == 'one_hot_string':
                    self.transformers.append(functools.partial(feature_utils.fix_na_one_hot_string, column=column))

            self.transformers.append(functools.partial(feature_utils.fill_na))

        """Handle lower-casing feature name and copying feature to new feature names"""
        for input_field, features in self.input_to_feature_map.items():
            for feature in features:
                # rename or copy transforms
                if input_field != feature.name:
                    # different case or only one mapping from input to feature
                    if input_field.lower() == feature.name.lower() or len(features) == 1:
                        # column_rename_map[input_field] = feature.name
                        self.transformers.append(functools.partial(feature_utils.rename_column, src_column=input_field, target_column=feature.name))
                    else:
                        # we need to copy the value
                        self.transformers.append(functools.partial(feature_utils.copy_value, src_column=input_field, target_column=feature.name))

        """Handle value transformations"""
        for input_field, features in self.input_to_feature_map.items():
            for feature in features:
                if isinstance(feature, CategoricalFeature):
                    # categorical features
                    if feature.value_to_number_mapper is not None:
                        # hash the value
                        if isinstance(feature.value_to_number_mapper, XXHashMapper):
                            self.transformers.append(functools.partial(feature_utils.xxhash_categorical_values_to_number,
                                                                       column=feature.name,
                                                                       value_to_lower=feature.transform_value_to_lowercase))
                        elif not isinstance(feature.value_to_number_mapper, NoneMapper):
                            raise ValueError(f"Unsupported CategoricalValueToNumberMapper for feature: {feature.name} - {feature.value_to_number_mapper}")
                    elif feature.transform_value_to_lowercase:
                        self.transformers.append(functools.partial(feature_utils.transform_value_to_lower, column=feature.name))
                elif isinstance(feature, LatLongFeature):
                    # lat-lang
                    self.transformers.append(functools.partial(feature_utils.box_lat_long_feature, column=feature.name))
                elif isinstance(feature, OneHotStringFeature):
                    # app_on_device feature
                    self.transformers.append(functools.partial(feature_utils.create_array_one_hot_feature, column=feature.name))

        # Handle categorical history features. This assumes that the categorical features are handled(hashed) because we might want to remove the label id
        # from the history to make sure that there is no leakage.
        for input_field, features in self.input_to_feature_map.items():
            for feature in features:
                if isinstance(feature, CategoricalHistoryFeature):
                    hash_ids = True if isinstance(feature.value_to_number_mapper, XXHashMapper) else False
                    self.transformers.append(functools.partial(
                        feature_utils.handle_categorical_history_feature,
                        column=feature.name,
                        hash_ids=hash_ids,
                        history_length=feature.history_length,
                        history_id_feature_name=feature.history_id_feature_name,
                        remove_history_id_from_history=feature.remove_history_id_from_history
                    ))

    def get_dtypes(self) -> dict[str, str]:
        return self.dtypes

    def get_input_columns(self) -> list[str]:
        return self.input_columns

    def get_input_to_feature_map(self) -> dict[str, list[Feature]]:
        return self.input_to_feature_map

    def get_features_map(self) -> dict[str, Feature]:
        return self.features_map

    def get_tensor_feature(self, key) -> Optional[TensorFeature]:
        feature = self.features_map.get(key)
        if feature is not None and feature.kind == FeatureKind.Tensor and isinstance(feature, TensorFeature):
            return feature
        else:
            return None

    def get_tensor_list_feature(self, key) -> Optional[TensorListFeature]:
        feature = self.features_map.get(key)
        if feature is not None and feature.kind == FeatureKind.TensorList and isinstance(feature, TensorListFeature):
            return feature
        else:
            return None

    def get_categorical_history_feature(self, key) -> Optional[CategoricalHistoryFeature]:
        feature = self.features_map.get(key)
        if feature is not None and feature.kind == FeatureKind.CategoricalHistory and \
                isinstance(feature, CategoricalHistoryFeature):
            return feature
        else:
            return None
    
    def get_one_hot_string_feature(self, key) -> Optional[OneHotStringFeature]:
        feature = self.features_map.get(key)
        if feature is not None and feature.kind == FeatureKind.OneHotString and \
                isinstance(feature, OneHotStringFeature):
            return feature
        else:
            return None

    def is_do_not_convert_to_platform_type(self, key) -> bool:
        feature = self.features_map.get(key)
        return feature is not None and feature.do_not_convert_to_platform_type

    def get_transformers(self) -> list[Callable[[pd.DataFrame], None]]:
        return self.transformers

    def default_data_mapper(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch = batch.astype(self.dtypes_string_map)

        # apply all transformers
        for transformer in self.transformers:
            transformer(batch)

        return batch

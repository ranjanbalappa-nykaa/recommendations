import enum
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class FileSystemKind(str, enum.Enum):
    LOCAL = "local"
    DBFS = "dbfs"
    S3 = "s3"



class FileSystemConfig(BaseModel):
    kind: FileSystemKind
    path_template: Optional[str] = None

    # dbfs fs params
    dbfs_base: Optional[str] = None

    #S3 fs params
    s3_bucket_path: Optional[str] = None


    # Local fs params
    local_dir_prefix: Optional[str] = None
    local_path_template: Optional[str] = None

    def __init__(self, **kwargs):
        filesystem = kwargs["kind"]
        if filesystem == FileSystemKind.DBFS:
            assert kwargs.get("dbfs_base", None) is not None, "dbfs_base must be specified for DBFS filesystem"

        elif filesystem == FileSystemKind.S3:
            assert kwargs.get("s3_bucket_path", None) is not None, "s3_bucket_path must be specified for S3 filesystem"

        elif filesystem == FileSystemKind.LOCAL:
            assert kwargs.get("local_dir_prefix", None) is not None, "local_dir_prefix must be specified for local filesystem"
            raise NotImplementedError("Currently local file system isn't supported")
        else:
            raise ValueError(f"Unsupported filesystem: {filesystem}")
        super().__init__(**kwargs)


class TrainDatasetConfig(BaseModel):
    filesystem_config: FileSystemConfig
    exclude_dates: List[str]
    train_data_ratio: float
    val_data_ratio: float
    extra_day_val_data_ratio: float = 1
    train_data_end_date: str
    train_period_in_days: int
    val_data_start_date: str
    val_period_in_days: int
    extra_day_val_data_start_date: Optional[str] = None
    extra_day_val_period_in_days: int = 1
    path_glob_train: str = ""
    path_glob_test: str = ""


class ModelInferenceConfig(BaseModel):
    num_workers: int
    max_num_batches: Optional[int] = None
    skip_inference: bool = False
    inference_batch_size: int


class ModelEvalConfig(BaseModel):
    num_workers: int

    # Eval (includes feature importance)
    skip_eval: bool = False
    eval_batch_size: int
    predict: bool = False
    compute_feature_importance: bool = False
    feature_importance_steps: int = 1
    max_eval_steps: int

    # KNN eval
    skip_knn_eval: bool = True
    knn_top_k_list: List[int] = [1, 5, 10, 20, 100, 200]
    knn_max_query_batches_per_worker: Optional[int] = None

    # Used when inference results are already present
    inference_results_path: Optional[str] = None


class ModelExportConfig(BaseModel):
    trace: bool = False
    filesystem_config: FileSystemConfig
    path_prefix: str
    export_config_str: bool
    export_inference_config: bool = False
    export_index_config: bool = False
    export_if_loss_within_factor_of_best_model: Optional[float] = None
    best_model_after_k_steps: Optional[int] = None


class ModelTrainConfig(BaseModel):
    num_workers: int
    use_gpu: bool
    batch_size: int
    train_steps: int
    validation_steps: int
    epochs: int
    learning_rate: float = 0.001
    train_metrics_every_n_steps: int
    val_metrics_every_n_steps: int
    gradient_clip_norm: Optional[float] = None
    gradient_clip_value: Optional[float] = None
    sparse_learning_rate: float = 0.25
    supress_ray_output_allreduce: bool = True
    weight_decay: Optional[float] = None
    # Takes precedence over learning_rate and weight_decay params
    optimizer_clazz: Optional[str] = None
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    # only available for all_reduce strategy
    lr_scheduler_clazz: Optional[str] = None
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None
    lr_scheduler_step_size: int = 100
    gradient_accumulation_steps: Optional[int] = None
    skip_train: bool = False
    checkpoint_every_k_steps: Optional[int] = None
    cache_every_k_val_batch: int = 40
    distributed_process_group_timeout_s: int = 1800
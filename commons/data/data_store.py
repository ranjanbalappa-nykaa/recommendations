from typing import Optional
from datetime import timedelta
from typing import List
import io
import logging
import os
import pandas as pd
from dateutil.parser import parse
import numpy as np
from abc import ABC, abstractmethod
import shutil
from overrides import override
import ray
import boto3
from retrying import retry
from pyarrow.parquet import read_table
from pyarrow import fs
from pyarrow import parquet as pq



from commons.configs.trainer_config import FileSystemConfig, FileSystemKind


def get_date_range_str(date: Optional[str], steps: Optional[int], backward: bool):
    if not date:
        return None
    dates = [parse(date) + (-1 if backward else 1) *
             timedelta(days=step) for step in range(steps)]
    return [d.strftime('%Y%m%d') for d in (dates[::-1] if backward else dates)]


def sample_paths(paths, data_ratio):
    arr = np.array(paths)
    np.random.shuffle(arr)
    data_ratio = 1 if data_ratio < 0 else data_ratio
    return arr[:int(data_ratio * len(arr))].tolist()


def is_valid_data_path(file_path: str) -> bool:
    return not file_path.endswith("_SUCCESS")


def traverse_dir_recursive(root: str) -> List[str]:
    results = []
    for root, subdir, files in os.walk(root):
        for f in files:
            results.append(os.path.join(root, f))
    return results

class DataStoreInterface(ABC):

    def __init__(self):
        # Disable polluting logs
        for path in [
            "azure.core.pipeline.policies.http_logging_policy",
            'azure.identity._credentials',
        ]:
            logger = logging.getLogger(path)
            logger.setLevel(logging.WARNING)

    def download_file_to_local_path(self, path: str, local_path: str):
        pass

    @abstractmethod
    def get_relative_paths_in_folder(self, folder_path: str) -> List[str]:
        ...

    @abstractmethod
    def get_training_data_paths_direct(self, direct_path: str) -> List[str]:
        ...

    @abstractmethod
    def get_training_data_paths_for_dates(self, data_dates: List[str], data_ratio: float = 1) -> List[str]:
        ...

    @abstractmethod
    def upload_dir_recursive(self, local_directory: str, folder=None) -> None:
        ...

    @abstractmethod
    def get_file_from_path(self, path: str) -> bytes:
        """Path here should be relative path from root of the filesystem
        """
        ...

    @abstractmethod
    def read_single_parquet_file(self, path: str, columns: Optional[List[str]]) -> pd.DataFrame:
        ...

    @abstractmethod
    def upload_byte_file(self, path: str, data: bytes) -> None:
        ...

class DataStoreAccessor:
    @staticmethod
    def get_instance(fs_config: FileSystemConfig) -> DataStoreInterface:
        if fs_config.kind == FileSystemKind.S3:
            return S3DataStore(s3_bucket=fs_config.s3_bucket_path, s3_template=fs_config.path_template)
        
        elif fs_config.kind == FileSystemKind.DBFS:
            return DBFSDataStore(dbfs_base=fs_config.dbfs_base, path_template=fs_config.path_template)


#s3 data store accessor
class S3DataStore(DataStoreInterface):
    def __init__(self, s3_bucket: str, s3_template: Optional[str]) -> None:
        super().__init__()
        self.s3_bucket = s3_bucket
        self.template = s3_template

    @ray.remote(scheduling_strategy="SPREAD")
    def _get_objects_for_hour(self, dt: str = "", hr: int = 0) -> List[str]:
        s3_loc = self.template.format(year=dt[:4], month=dt[4:6], day=dt[6:], hr=hr)
        s3 = boto3.resource('s3')
        s3_bucket_resc = s3.Bucket(self.s3_bucket)
        data_paths = []

        try:
            for object_summary in s3_bucket_resc.objects.filter(Prefix=s3_loc):
                if is_valid_data_path(object_summary.key):
                    data_paths.append(object_summary.key)
        except Exception as e:
            print("Error in running s3_client: ", e)
        return data_paths

    @ray.remote(scheduling_strategy="SPREAD")
    def _get_objects_for_date(self, dt: str = "") -> List[str]:
        s3_loc = self.template.format(date=dt)
        s3_client = boto3.client('s3', region_name="ap-south-1")
        s3_bucket_resc = s3_client.Bucket(self.s3_bucket)
        data_paths = []

        try:
            for object_summary in s3_bucket_resc.objects.filter(Prefix=s3_loc):
                if is_valid_data_path(object_summary.key):
                    data_paths.append(object_summary.key)
        except Exception as e:
            print("Error in running s3_client: ", e)

        # paths = []
        # futures = []
        # for hr in range(24):
        #     s3_path = self.template.format(date=dt)
        #     if s3_path in paths:
        #         continue
        #     paths.append(s3_path)
        #     try:
        #         if 'CommonPrefixes' in s3_client.list_objects(
        #                 Bucket=self.s3_bucket, Prefix=s3_path, Delimiter='/', MaxKeys=1):
        #             futures.append(self._get_objects_for_hour.remote(self, dt, hr))
        #     except Exception as e:
        #         print("Error in running s3_client: ", e)

        # data_paths = []
        # for paths in ray.get(futures):
        #     data_paths.extend(paths)

        # return data_paths
        return data_paths
      

    def _get_objects_for_dates(self, data_dates: List[str]) -> List[str]:
        futures = []
        for dt in data_dates:
            futures.append(self._get_objects_for_date.remote(self, dt))
        data_paths = []
        for paths in ray.get(futures):
            data_paths.extend(paths)
        return data_paths

    @override
    def download_file_to_local_path(self, path: str, local_path: str):
        boto3.client('s3').download_file(self.s3_bucket, path, local_path)

    @override
    def get_relative_paths_in_folder(self, folder_path: str) -> List[str]:
        s3_bucket_resc = boto3.resource('s3').Bucket(self.s3_bucket)
        data_paths = []
        try:
            for object_summary in s3_bucket_resc.objects.filter(Prefix=folder_path):
                data_paths.append(object_summary.key)
        except Exception as e:
            print("Error in running s3_client: ", e)
        return data_paths

    @override
    def get_training_data_paths_for_dates(
            self,
            data_dates: List[str],
            data_ratio: float = 1
    ) -> List[str]:
        if self.template is None:
            raise ValueError("get_training_data_paths_for_dates doesn't work with empty template, please provide template value while data store init")
        data_paths = self._get_objects_for_dates(data_dates)
        return sample_paths(data_paths, data_ratio)

    @override
    def get_training_data_paths_direct(self, direct_path: str) -> List[str]:
        result = self.get_relative_paths_in_folder(direct_path)
        result = [f"{self.s3_bucket}/{relative_path}" for relative_path in result]
        return list(filter(lambda path: is_valid_data_path(path), result))

    @override
    def get_file_from_path(self, path: str) -> bytes:
        print(f"Reading file from s3: {path}")
        client = boto3.client('s3')
        response = client.get_object(Bucket=self.s3_bucket, Key=path)
        return response['Body'].read()

    @override
    def upload_dir_recursive(self, local_directory: str, folder=None):
        # Helper method with retries
        @retry(
            stop_max_attempt_number=10,
            wait_exponential_multiplier=1000,   # start with 1s
            wait_exponential_max=60000,         # cap at 60s (optional)
            wait_random_min=1000,               # jitter: +1s
            wait_random_max=2000                # jitter: +2s
        )
        def _upload(_client, _local_path, _bucket, _s3_path):
            _client.upload_file(_local_path, _bucket, _s3_path)

        client = boto3.client('s3')
        # Remove the leading '/' and replace tmp by models folder
        s3_folder = local_directory[1:].replace("tmp", "models") if folder is None else folder

        local_files = traverse_dir_recursive(local_directory)
        for local_path in local_files:
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_folder, relative_path)
            print(f"Uploading {local_path} to {self.s3_bucket}/{s3_path}...")
            _upload(client, local_path, self.s3_bucket, s3_path)
            print(f"Done uploading {local_path} to {self.s3_bucket}/{s3_path}...")

    # FIXME transient curl errors in AWS s3 not handled by pyarrow
    #  see https://lists.apache.org/thread/v7v2h3439pxj9konzncjd2hw542nhd05
    #  hence we wrap this block with retry logic
    @retry(
        stop_max_attempt_number=10,
        wait_exponential_multiplier=1000,   # start with 1s
        wait_exponential_max=60000,         # cap at 60s (optional)
        wait_random_min=1000,               # jitter: +1s
        wait_random_max=2000                # jitter: +2s
    )
    def read_single_parquet_file(self, path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if path.startswith('s3://'):
            raise ValueError("path should not start with s3://")
        table = read_table(f"{self.s3_bucket}/{path}", columns=columns, filesystem=fs.S3FileSystem(region="ap-south-1"))
        return table.to_pandas()

    def upload_byte_file(self, path: str, data: bytes) -> None:
        resource = boto3.resource('s3')
        s3_object = resource.Object(self.s3_bucket, path)
        s3_object.put(Body=data)

    
        


#Databricks
class DBFSDataStore(DataStoreInterface):
    def __init__(self, dbfs_base: str, path_template: str) -> None:
        super().__init__()
        # store both dbfs:/ and /dbfs/ forms
        self.dbfs_base = dbfs_base
        self.local_base = dbfs_base.replace("dbfs:/", "/dbfs/")
        self.path_template = path_template

    def _to_local_path(self, path: str) -> str:
        """Convert dbfs:/ path -> /dbfs/ path"""
        if path.startswith("dbfs:/"):
            return path.replace("dbfs:/", "/dbfs/")
        return path

    def _to_dbfs_path(self, path: str) -> str:
        """Convert /dbfs/ path -> dbfs:/ path"""
        if path.startswith("/dbfs/"):
            return path.replace("/dbfs/", "dbfs:/")
        return path
    
    def get_relative_paths_in_folder(self, folder_path: str) -> List[str]:
        local_folder = self._to_local_path(folder_path)
        data_paths = []
        try:
            for root, dirs, files in os.walk(local_folder):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), local_folder)
                    data_paths.append(rel)
        except Exception as e:
            print(f"Error listing DBFS folder {folder_path}: {e}")
        return data_paths

    def get_training_data_paths_for_dates(
        self,
        data_dates: List[str],
        data_ratio: float = 1
    ) -> List[str]:
        # For DBFS, template-based hour/date expansion is optional.
        # Assume `data_dates` are folders under base.
        all_paths = []
        for dt in data_dates:
            folder = os.path.join(self.local_base, dt)
            if os.path.exists(folder):
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        full_path = os.path.join(root, f)
                        if is_valid_data_path(full_path):
                            all_paths.append(full_path)
        return sample_paths(all_paths, data_ratio)

    def get_training_data_paths_direct(self, direct_path: str) -> List[str]:
        local_path = self._to_local_path(direct_path)
        all_files = []
        for root, dirs, files in os.walk(local_path):
            for f in files:
                full_path = os.path.join(root, f)
                if is_valid_data_path(full_path):
                    all_files.append(full_path)
        return all_files

    def download_file_to_local_path(self, path: str, local_path: str):
        local_src = self._to_local_path(path)
        shutil.copy(local_src, local_path)

    def get_file_from_path(self, path: str) -> bytes:
        local_src = self._to_local_path(path)
        with open(local_src, "rb") as f:
            return f.read()
        
    def upload_dir_recursive(self, local_directory: str, folder: Optional[str] = None):
        target_dir = os.path.join(self.local_base, folder) if folder else self.local_base
        os.makedirs(target_dir, exist_ok=True)
        for local_file in traverse_dir_recursive(local_directory):
            relative_path = os.path.relpath(local_file, local_directory)
            dbfs_dest = os.path.join(target_dir, relative_path)
            os.makedirs(os.path.dirname(dbfs_dest), exist_ok=True)
            shutil.copy(local_file, dbfs_dest)
            print(f"Uploaded {local_file} -> {dbfs_dest}")


    def read_single_parquet_file(self, path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        local_path = self._to_local_path(path)
        return pd.read_parquet(local_path, columns=columns)

    def upload_byte_file(self, path: str, data: bytes) -> None:
        local_path = self._to_local_path(path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
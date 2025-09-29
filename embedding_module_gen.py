from argparse import ArgumentParser

import ray
import ast
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime, timedelta
import pandas as pd
import tempfile
import os

from commons.data.data_store import DataStoreAccessor
import pickle
import io

from commons.layers import MLP, KShiftEmbedding
from commons.feature_utils import hash_feature_name_to_int, hash_string_to_long, MAX_LONG_VALUE_PLUS_ONE
from commons.configs.trainer_config import FileSystemConfig


parser = ArgumentParser()





class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, mask_model: nn.Module):
        super().__init__()
        self.model = model
        self.mask_model = mask_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model(x)
        mask = self.mask_model(x).sigmoid()
        return mask * emb
    

def get_product_embeddings(data_store, args):
    df = data_store.read_single_parquet_file(args.emb_base_folder)
    df['embedding'] = df['emb_128'].apply(lambda x: x[:args.dim])
    df = df[['product_id', 'embedding']]
    return df


    

def massage_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): _description_
    """
    seed = hash_feature_name_to_int('product_id')

    def hash_(id_):
        return hash_string_to_long(id_, seed, value_to_lower=False)

    df['product_id'] = df['product_id'].apply(hash_).values.astype(np.int64)
    print(f'embeddings count is {df.shape[0]}!!')

    return df


#train mask model
def train_mask_model(
        df: pd.DataFrame,
        expansion_factor: float,
        k_shift: int,
        mask_emb_dim: int,
) -> nn.Module:
    product_id = torch.from_numpy(df['product_id'].values.astype(np.int64))
    n = product_id.size(0)

    model = nn.Sequential(
        KShiftEmbedding(
            int(expansion_factor * n),
            mask_emb_dim,
            num_shifts=k_shift,
            normalize_output=False,
        ),
        MLP(mask_emb_dim, 1, [mask_emb_dim * 16]),
    )

    batch_size = 2 ** 17
    num_epochs = 100

    num_batches = n // batch_size
    if num_batches * batch_size < n:
        num_batches += 1
    idx_true = np.arange(0, n)

    optim = torch.optim.Adagrad(model.parameters(), lr=5e-1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        idx_reordered = np.array(idx_true)
        np.random.shuffle(idx_reordered)

        for batch_nb in range(num_batches):
            idx_this = idx_reordered[(batch_nb * batch_size): min((batch_nb + 1) * batch_size, n)]
            product_id_pos = product_id[idx_this]
            product_id_neg = torch.randint(-MAX_LONG_VALUE_PLUS_ONE, MAX_LONG_VALUE_PLUS_ONE - 1,
                                        (product_id_pos.size(0),), dtype=torch.int64)
            product_id_this = torch.cat([product_id_pos, product_id_neg], dim=0)
            target = torch.cat([torch.ones_like(product_id_pos), torch.zeros_like(product_id_neg)], dim=0)
            prediction = model(product_id_this).squeeze(1)
            loss = criterion(prediction, target.float())
            loss.backward()
            optim.step()
            optim.zero_grad()
            print('MASK', epoch, num_epochs, batch_nb, num_batches, loss.detach().item())

    return model


#train reconstruction model
def train_model(df: pd.DataFrame, expansion_factor: float, k_shift: int) -> nn.Module:
    hashed_idx = torch.from_numpy(df['product_id'].values.astype(np.int64))
    x = torch.from_numpy(np.stack(df['embedding'].values).astype(np.float32))
    x = F.normalize(x, p=2.0, dim=-1)
    model = KShiftEmbedding(int(expansion_factor * x.size(0)), x.size(1), num_shifts=k_shift, normalize_output=True)

    batch_size = 2 ** 18
    num_epochs = 500

    num_batches = hashed_idx.shape[0] // batch_size
    if num_batches * batch_size < hashed_idx.shape[0]:
        num_batches += 1

    idx_true = np.arange(0, x.size(0))

    optim = torch.optim.Adagrad(model.parameters(), lr=5e-1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        idx_reordered = np.array(idx_true)
        np.random.shuffle(idx_reordered)

        for batch in range(num_batches):
            idx_this = idx_reordered[batch * batch_size : min((batch + 1) * batch_size, x.size(0))]
            input_this = hashed_idx[idx_this]
            x_this = x[idx_this].float()
            optim.zero_grad()
            y_this = model(input_this)
            batch_size = x_this.size(0)
            loss = criterion(y_this, x_this)
            loss.backward()
            optim.step()
            print('Model', epoch, num_epochs, batch, num_batches, loss.detach().item())

    return model




@ray.remote(num_cpus=8)
def execute(args):
    data_store = DataStoreAccessor.get_instance(
        FileSystemConfig(
            kind="s3",
            s3_bucket_path=args.s3_bucket,
        )
    )

    #get embeddings
    df = get_product_embeddings(data_store, args)


    df_ = massage_embeddings(df)
    model = train_model(
        df_,
        args.expansion_factor,
        args.k_shift,
    )
    mask_model = train_mask_model(
        df_,
        args.expansion_factor,
        args.k_shift,
        args.mask_emb_dim,
    )
    final_model = ModelWrapper(model, mask_model)

    with tempfile.TemporaryDirectory() as tmp:
        final_model.eval()
        path = os.path.join(tmp, f"model_scripted_0.pt")
        model_ = torch.jit.script(final_model)
        torch.jit.save(
            model_,
            path
        )
        data_store.upload_dir_recursive(local_directory=tmp, folder=os.path.join(args.s3_base_folder, args.dt))
        print(f"Model saved to {path}")





def main(args):
    print(f"Received args : {args}")
    try:
        ray.init()
        ray.get(execute.remote(args))
    finally:
        ray.shutdown()




if __name__ == "__main__":
    parser.add_argument(
        "--dt",
        required=True,
        type=str,
        help=f"date",
    )
    parser.add_argument(
        "--s3_bucket",
        default='',
        type=str,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3_base_folder",
        default='ranjan.balappa/aux_emb/dev',
        type=str,
        help="Base S3 folder name",
    )
    parser.add_argument(
        "--emb_base_folder",
        default='ranjan.balappa/aux_emb/dev',
        type=str,
        help="Base S3 folder name",
    )
 
    parser.add_argument(
        '--dim',
        default=32,
        type=int,
        help="embedding dim"
    )
 
    parser.add_argument(
        '--expansion_factor',
        default=1.15,
        type=float,
        help="how many extra params to use"
    )
    parser.add_argument(
        '--k_shift',
        default=16,
        type=int,
        help="how many shifts to use"
    )
    parser.add_argument(
        '--mask_emb_dim',
        default=4,
        type=int,
        help="how many dims to use for mask prediction"
    )

    main(parser.parse_args())
 
    
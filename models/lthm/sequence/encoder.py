from typing import Dict

import io
import os

import torch
import torch.nn as nn

from lthm.config import LTHMModelConfig
from lthm.sequence.query_tower import QueryTower
from lthm.sequence.product_tower import ProductTower

from commons.layers import QREmbedding, KShiftEmbedding 
from commons.data.data_store import DataStoreAccessor



class Encoder(nn.Module):
    def __init__(self, model_config: LTHMModelConfig):
        super().__init__()
        product_tower_config = model_config.product_tower


        #If the product embedding is already generated using any encoder enclode ids, embedding in torchscript
        if product_tower_config.model_init_metadata is not None:
            model_init_metadata = product_tower_config.model_init_metadata
            data_store = DataStoreAccessor.get_instance(model_init_metadata.filesystem_config)
            buffer = io.BytesIO(data_store.get_file_from_path(model_init_metadata.embedding_module_path))
            self.product_emb_module = torch.jit.load(buffer)

        else:
            self.product_emb_module = KShiftEmbedding(
                product_tower_config.latent_model_config.vocab_size_latent,
                product_tower_config.out_emb_dim,
                num_shifts=product_tower_config.latent_model_config.num_shifts_latent,
                normalize_output=product_tower_config.latent_model_config.normalize_embedding,
            )

        self.product_tower = ProductTower(model_config)
        self.query_tower = QueryTower(model_config)



    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ids = batch['product_ids']
        embs = self.product_emb_module(ids)

        #get product projection and embedding
        inp, target, mask = self.product_tower(ids, embs)

        #flip all the inputs to have left padding
        inp, target, mask, labels, timestamp , ids = self.flip_all(
            inp, target, mask, batch["labels"], batch["timestamp"], ids
        )
        
        output = self.query_tower(inp, target, mask, labels, timestamp, ids)

        return output
    
    def flip_all(self, *tensors):
        return [torch.flip(t, dims=[1]) for t in tensors]

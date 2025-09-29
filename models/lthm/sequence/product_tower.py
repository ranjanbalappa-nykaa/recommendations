from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from commons.transformers.layers import CosineVectorEmbedding
from commons.layers import HistogramEmbedding
from models.lthm.config import LTHMModelConfig


class ProductTower(nn.Module):
    def __init__(self, model_config: LTHMModelConfig):
        super().__init__()
        tower_config = model_config.product_tower
        self.inp_emb_dim = tower_config.inp_emb_dim
        self.out_emb_dim = tower_config.out_emb_dim
        self.norm_threshold = tower_config.norm_threshold
        self.norm_bins = tower_config.norm_bins

        self.emb_mapper = nn.Linear(tower_config.inp_emb_dim, tower_config.out_emb_dim)

        self.direction_emb = nn.ModuleList([
            CosineVectorEmbedding(
                tower_config.inp_emb_dim,
                tower_config.out_emb_dim,
                num_proj=config.num_proj,
                num_bins=config.num_bins
            )
            for config in tower_config.cosine_lsh_config
        ])

        if self.norm_bins > 1:
            self.norm_emb = HistogramEmbedding(
                0, 1, tower_config.norm_bins,
                emb_dim=tower_config.out_emb_dim,
            )

        self.product_mapper = nn.Linear(
            tower_config.out_emb_dim, 
            tower_config.product_emb_dim, 
            bias=False
        )

    def forward(self, ids: torch.Tensor, x: torch.Tensor):
        bsz, seq_len = ids.shape
        device = ids.device

        x = x.detach()   # stop gradients
        x_norm = x.norm(p=2.0, dim=-1)
        mask = torch.logical_or(x_norm < self.norm_threshold, ids == 0)

        x = F.normalize(x, p=2.0, dim=-1)
        emb = self.emb_mapper(x)
        for mod in self.direction_emb:
            emb = emb + mod(x)

        if self.norm_bins > 1:
            emb = emb + self.norm_emb(x_norm)

        emb = emb.masked_fill(mask.unsqueeze(-1), 0.0)
        prod_emb = self.product_mapper(emb)

        return emb, prod_emb, mask

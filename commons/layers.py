import math

import torch
import torch.nn as nn 
from torch import Tensor 
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class PatternFromTimelocal(nn.Module):
    """
    Extracts patterns from timestamp_local (in epoch time)
    Usage:
        DayOfWeek
            div = 60 * 60 * 24
            mod = 7
        HourOfWeek
            div = 60 * 60
            mod = 7 * 24
        HourOfDay
            div = 60 * 60 
            mod = 24
    """
    def __init__(self, div, mod, emb_dim):
        self.div = div
        self.mod = mod
        self.emb_dim = emb_dim

        if self.emb_dim > 0:
            self.emb = nn.Embedding(num_embeddings=mod, emb_dim=emb_dim)

        else:
            self.emb = nn.Identity()

    def forward(self, x: Tensor):
        index = torch.remainder(torch.floor_divide(x.long(), self.div), self.mod)
        return self.emb(index)
    

class FlatEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, emb_dim: int, padding_idx: int = None, 
                 zero_init: bool = False, normalize_output: bool = False):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._emb_dim = emb_dim
        self.padding_idx = padding_idx
        self._emb_table = nn.Embedding(self._num_embeddings, self._emb_dim, padding_idx=self.padding_idx)
        self._normalize_output = normalize_output
        if zero_init:
            self._emb_table.weight.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.remainder(x, self._num_embeddings).long()
        x = self._emb_table(x)
        if self._normalize_output:
            x = F.normalize(x, p=2.0, dim=-1)
        return x
    


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, gate_sizes):
        super().__init__()
        previous_dim = input_dim
        blocks = []
        for gate_size in gate_sizes:
            blocks.append(
                nn.Linear(previous_dim, gate_size),
            )
            blocks.append(QuickGELU())
            previous_dim = gate_size

        blocks.append(nn.Linear(previous_dim, out_dim))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.model(x)
    

class NAImputationPlusQuantileEmbedding(nn.Module):
    def __init__(self, na_value, quantiles, eps=1e-6):
        super().__init__()
        self.na_value = na_value
        self.register_buffer("quantiles", torch.tensor(quantiles))
        self.emb = nn.Embedding.from_pretrained(
            torch.arange(0, len(quantiles) - 1, 1) / len(quantiles) - 0.5,
            freeze=False
        )
        self.eps = eps
        self.na_param = nn.Parameter(torch.zeros(1,))

    def forward(self, x):
        x = x.float()
        y = self.emb(torch.bucketize(x, self.quantiles))
        return torch.where((x - self.na_value) < self.eps, self.na_param.unsqueeze(-1), y)
    

class QREmbedding(nn.Module):
    """
    QR trick from ByteDance: https://arxiv.org/abs/2209.07663
    """
    def __init__(self, num_embeddings: int, emb_dim: int, normalize_output: bool):
        self._div = int(math.sqrt(num_embeddings))
        self.num_embeddings = self._div * self._div
        self.emb_dim = emb_dim
        self.emb_q = nn.Embedding(self._div, emb_dim)
        self.emb_r = nn.Embedding(self._div, emb_dim)
        self.normalize_output = normalize_output

    
    def forward(self, x):
        x = torch.remainder(x, self.num_embeddings)
        q = torch.remainder(torch.div(x, self._div, round_mode="floor"), self._div)
        r = torch.remainder(x, self._div)
        x = self.emb_q(q) + self.emb_r(r)
        if self.normalize_output:
            x = F.normalize(x, p=2.0, dim=-1)
        
        return x
    
class KShiftEmbedding(nn.Module):
    """
    See https://arxiv.org/abs/2207.10731 for high level motivation on why this may work
    """

    def __init__(
            self,
            num_embeddings: int,
            emb_dim: int,
            num_shifts: int = 8,
            normalize_output: bool = False,
            sparse: bool = False,
    ):
        """
        - num_embeddings: size of shared embedding table (P in paper, Sec 2, pg 4)
        - emb_dim: embedding dimension (d in paper)
        - num_shifts: number of hash functions / rotations (k in Sec 2.1, Theorem 2.2, pg 5)
        - normalize_output: optional L2 normalization (see discussion below Fig. 3, pg 8)
        - sparse: whether to use sparse gradients (implementation detail)
        """
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim, sparse=sparse)
        self._num_embeddings = num_embeddings
        self._num_shifts = num_shifts
        self._num_bits = 64   # input IDs assumed to be 64-bit integers
        self._normalize_output = normalize_output

    def forward(self, id_: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass = parameter shared setup (Sec 2, pg 4):
        Each ID -> multiple row indices -> summed embeddings -> scaled/normalized.
        """
        # First lookup (col_idx = 0, direct remainder hash)
        idx = self.get_row_idx(id_, 0)
        x = self.emb(idx)

        # Subsequent lookups with bit-rotations (Sec 3.2 "QR embeddings", pg 6–7)
        for col_idx in range(1, self._num_shifts):
            idx = self.get_row_idx(id_, col_idx)
            x = x + self.emb(idx)

        # Either L2 normalize (Fig 3 caption, pg 8) or scale (JL scaling, Theorem 2.2, pg 5)
        if self._normalize_output:
            x = F.normalize(x, p=2.0, dim=-1)
        else:
            x = x / math.sqrt(self._num_shifts)

        return x

    def get_row_idx(self, x: torch.LongTensor, col_idx: int):
        """
        Generate pseudo-random row index (mapping M in Sec 2.1, pg 4).
        - col_idx=0 → base hash via modulus
        - col_idx>0 → apply bit rotation (rotation-based QR embedding, Sec 3, pg 6–7)
        """
        if col_idx != 0:
            # Mirrors bit rotation operator (like QR trick, Sec 3.2)
            x = (x << col_idx) | (x >> (self._num_bits - col_idx))

        # Modulo to map into shared parameter table (hashed lookup, Sec 3.1, pg 6)
        return torch.remainder(x, self._num_embeddings)

    

class StreamingLogQCorrectionModule(nn.Module):
    """
    See https://research.google/pubs/pub48840 for an overview of this method
    """
    def __init__(self, num_buckets, hash_offset, alpha: float=0.05, p_init: float=0.01):
        super().__init__()
        self.num_buckets = num_buckets
        self.hash_offset = hash_offset
        self.alpha = alpha
        self.p_init = p_init
        self.register_buffer("b", (1.0 / p_init) * torch.ones((num_buckets,), dtype=torch.float32))
        self.register_buffer("a", torch.zeros((num_buckets,), dtype=torch.float))

    def forward(self, products: torch.Tensor) -> torch.Tensor:
        h = self.hash_fn(products)
        return -self.b[h].log().reshape(*products.shape)
    
    def hash_fn(self, products):
        hash = (products + self.hash_offset) % self.num_buckets
        return hash
    
    def train_step(self, products: torch.Tensor, batch_idx: int):
        hash = self.hash_fn(products)
        self.b[hash] = ((1 - self.alpha) * self.b[hash]) + (self.alpha * (batch_idx - self.a[hash])).float()
        self.alpha[hash] = batch_idx

                                              

class CascadedStreamingLogQCorrectionModule(nn.Module):
    def __init__(self, num_buckets, hash_offsets, alpha: float=0.05, p_init: float=0.01):
        super().__init__()
        self.models = nn.ModuleList([
            StreamingLogQCorrectionModule(num_buckets, offset, alpha, p_init)
            for offset in hash_offsets
        ])

    def forward(self, products):
        result = torch.empty((0, ), device=products.device)
        for i, mod in enumerate(self.models):
            if i == 0:
                result = mod(products)
            else:
                result = torch.minimum(result, mod(products))

        return result
    
    def train_step(self, products, batch_idx):
        for mod in enumerate(self.models):
            mod.train_Step(products, batch_idx)




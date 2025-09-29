import math

import torch
import torch.nn as nn
from torch import Tensor
from commons.transformers.layers import TransformerBlock
from commons.layers import FlatEmbedding, PatternFromTimelocal
from models.lthm.config import LTHMModelConfig




class QueryTower(nn.Module):
    def __init__(self, model_config: LTHMModelConfig):
        super().__init__()
        transformer_config = model_config.transformer_config
        self.ememb_dim = emb_dim = model_config.emb_dim
        context_width = model_config.context_width

        #input projection
        self.inp_proj = nn.Linear(model_config.product_tower.out_emb_dim, emb_dim)

        #Action Embedding
        self.action_embedding = FlatEmbedding(4, emb_dim)

        #Time Embedding
        self.time_embedding = nn.ModuleDict(
            dict(
                hod=PatternFromTimelocal(60 * 60, 24, emb_dim),
                how=PatternFromTimelocal(60 * 60, 24 * 7, emb_dim),
                dow=PatternFromTimelocal(60 * 60 * 24, 7, emb_dim)
            )
        )


        #Transformer block
        self.transformer = nn.ModuleDict(
            dict(
                dropout=nn.Dropout(transformer_config.dropout),
                residual_attn=nn.ModuleList([
                        TransformerBlock(transformer_config, seed=depth)
                        for depth in range(transformer_config.num_layers)
                ])
            )
        )

        self.wpe = nn.Embedding(context_width + 1, emb_dim)
        self.pad = nn.Parameter(torch.randn((1, 1, emb_dim)) / math.sqrt(emb_dim))

        #MLP Heads
        self.export_tokens = model_config.export_tokens
        self.export_span = model_config.export_span
        self.outcome_conditioning = FlatEmbedding(4, emb_dim)
        self.emb_heads = nn.ModuleList([
            nn.Linear(emb_dim, model_config.product_tower.product_emb_dim, bias=False)
            for _ in range(self.export_tokens)
        ])


    def forward(
        self, 
        input: Tensor, 
        target: Tensor, 
        mask_inp: torch.BoolTensor, 
        labels: torch.LongTensor,
        timestamp: Tensor,
        ids: Tensor,
        future_outcome: Tensor = torch.zeros((1, 1)),
    ):
        bsz, orig_seq_len, emb_dim = input.size()
        device = input.device

        mask = mask_inp.unsqueeze(-1)
        mask_all_bs = mask.all(dim=0)
        if mask_all_bs.sum() > orig_seq_len - self.export_span:
            trim = torch.tensor([orig_seq_len - self.export_span], dtype=torch.int64)[0]
        else:
            trim_tensor = torch.nonzero(((~mask_all_bs).cumsum(dim=0, dtype=torch.long) > 0).squeeze(1)).squeeze(1)
            trim = trim_tensor[0]

        input = input[:, trim:].contiguous()
        mask = mask[:, trim:].contiguous()
        labels = labels[:, trim:].contiguous().long()
        timestamp = timestamp[:, trim:].contiguous().long()
        target = target[:, trim:].contiguous()
        ids = ids[:, trim:].contiguous()

        #action embeddfing
        emb_action = self.action_embedding(labels)
        

        #time embedding
        emb_hod = self.time_embedding.hod(timestamp)
        emb_how = self.time_embedding.how(timestamp)
        emb_dow = self.time_embedding.dow(timestamp)

        x = self.inp_proj(input) + emb_action + emb_hod + emb_how + emb_dow 
        seq_len = x.size(1)
        x = torch.where(
            mask,
            self.pad.expand(bsz, seq_len, -1),
            x
        )

        #left padding
        pos = seq_len - torch.arange(0, seq_len + 1, device=device).unsqueeze(0)
        x = torch.cat(
            (torch.zeros(1, 1, self.emb_dim, device=device).expand(bsz, -1, -1), x),
            dim=1
        )
        x = x + self.wpe(pos)
        
        
        #attention
        x = self.transformer_encoder(x)

        #outputmlp
        outcomes = torch.cat(
            (labels, future_outcome.to(device=device, dtype=torch.long).expand(bsz, -1)),
            dim=-1
        )
        x = x + self.outcome_conditioning(outcomes)
        y = torch.stack([mod(x) for mod in self.emb_heads], dim=2)

        return {
            "current_token_emb": target,
            "next_token_emb": y,
            "current_token_mask": mask,
            "current_token_ids": ids,
        }
    
    def transformer_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer.dropout(x)
        for mod in self.transformer.residual_attn:
            x = x + mod(x)

        return x


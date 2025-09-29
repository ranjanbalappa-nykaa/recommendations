
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# RelativePositionBias
# -------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, nq: int, nk: int, nh: int):
        super().__init__()
        self.nq = nq
        self.nk = nk
        self.nh = nh
        self.bias = nn.Parameter(torch.zeros((nq + nk + 1, nh)))

    def forward(self, qk: torch.Tensor) -> torch.Tensor:
        # qk: (..., nq, nk)
        nq = qk.size(-2)
        nk = qk.size(-1)
        device = qk.device
        if not (nq <= self.nq):
            raise RuntimeError("nq > self.nq")
        if not (nk <= self.nk):
            raise RuntimeError("nk > self.nk")
        pos_q = torch.arange(0, nq, device=device).unsqueeze(1)  # (nq,1)
        pos_k = torch.arange(0, nk, device=device).unsqueeze(0)  # (1,nk)
        pos_qk = pos_q - pos_k + nk  # (nq,nk)
        bias_qk = self.bias[pos_qk]  # (nq, nk, nh)
        bias_qk = bias_qk.permute(2, 0, 1).unsqueeze(0)  # (1, nh, nq, nk)
        return qk + bias_qk  # broadcast over leading dims


# -------------------------
# ScaledDotProductAttention
# -------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, nq: int, nk: int, nh: int, relative_bias: bool = False):
        super().__init__()
        if relative_bias:
            self.pos_bias = RelativePositionBias(nq, nk, nh)
        else:
            self.pos_bias = nn.Identity()

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query: (B, H, S_q, E), key: (B, H, S_k, E), value: (B, H, S_k, E)
        head_size = query.size(-1)
        qk = (query @ key.transpose(-2, -1)) / math.sqrt(float(head_size))  # (B,H,S_q,S_k)
        qk = self.pos_bias(qk)
        if mask is not None:
            qk = qk + mask
        qk = F.softmax(qk, dim=-1)
        return qk @ value  # (B,H,S_q,E)


# -------------------------
# Simple helper MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, gate_sizes: Optional[Tuple[int, ...]] = None, bias: bool = True):
        super().__init__()
        gate_sizes = gate_sizes if gate_sizes is not None else []
        blocks: List[nn.Module] = []
        prev = in_features
        for g in gate_sizes:
            blocks.append(nn.Linear(prev, g, bias=bias))
            blocks.append(nn.GELU(approximate="tanh"))
            prev = g
        blocks.append(nn.Linear(prev, out_features, bias=bias))
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -------------------------
# _MoEUnit used by MoELinear
# -------------------------
class _MoEUnit(nn.Module):
    def __init__(self, in_features: int, out_features: int, proj_features: int):
        super().__init__()
        self.l1 = nn.Linear(in_features, proj_features)
        self.activation = nn.GELU(approximate="tanh")
        self.l2 = nn.Linear(proj_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.activation(self.l1(x)))


# -------------------------
# MoELinear (Torch-friendly)
# -------------------------
class MoELinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 proj_features: int,
                 num_experts: int,
                 bias: bool = True,
                 top_k: Optional[int] = None,
                 gate_sizes: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.expert_gates = MLP(in_features, num_experts, gate_sizes=gate_sizes, bias=bias)
        experts: List[nn.Module] = []
        for _ in range(num_experts):
            experts.append(_MoEUnit(in_features, out_features, proj_features))
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        gate_values = self.expert_gates(x) / math.sqrt(float(self._in_features))  # (..., num_experts)
        if self.top_k is not None:
            k = min(self.top_k, gate_values.size(-1))
            v, _ = torch.topk(gate_values, k, dim=-1, largest=True, sorted=True)
            thresh = v[..., -1].unsqueeze(-1)
            gate_values = torch.where(gate_values < thresh, torch.tensor(-float('inf'), device=gate_values.device), gate_values)
        gate_values = F.softmax(gate_values, dim=-1)  # (..., num_experts)

        outs: List[torch.Tensor] = []
        for mod in self.experts:
            outs.append(mod(x))  # (..., out_features)
        expert_outputs = torch.stack(outs, dim=-2)  # (..., num_experts, out_features)
        gates = gate_values.unsqueeze(-1)  # (..., num_experts, 1)
        out = (expert_outputs * gates).sum(dim=-2)  # (..., out_features)
        return out


# -------------------------
# LayerNorm helpers
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class LayerNormND(nn.Module):
    def __init__(self, shape: Tuple[int, ...], bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*shape))
        self.bias = nn.Parameter(torch.zeros(*shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


# -------------------------
# SelfAttention base + MultiQuery/MultiHead
# -------------------------
class SelfAttentionConfig:
    # lightweight config object for type hints; in practice supply attributes
    def __init__(self, n_embd: int, n_head: int, attn_dropout: float, dropout: float, bias: bool,
                 pos_bias: Optional[Any] = None):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.bias = bias
        self.pos_bias = pos_bias


class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        if config.pos_bias is None:
            self.attn = ScaledDotProductAttention(0, 0, 0, relative_bias=False)
        else:
            self.attn = ScaledDotProductAttention(
                nq=config.pos_bias.context_window,
                nk=config.pos_bias.context_window,
                nh=config.n_head,
                relative_bias=True,
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError("Use subclass")

    @classmethod
    def from_config(cls, config: SelfAttentionConfig):
        # For simplicity, choose multi-head as default when available
        # A production config enum selector can be added
        return MultiHeadAttention(config)


class MultiQueryAttention(SelfAttention):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__(config)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # kv projection reduces last dim so that kv has size 2 * (n_embd // n_head)
        self.kv_proj = nn.Linear(config.n_embd, 2 * (config.n_embd // config.n_head), bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, t, _ = x.size()
        device = x.device
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        # split kv into k and v along last dim
        k, v = kv.split(self.n_embd // self.n_head, dim=-1)

        ones = torch.ones((bs, 1, t, 1), device=device)
        k_do = self.attn_dropout(ones)
        q_do = self.attn_dropout(ones)
        v_do = self.attn_dropout(ones)

        q = q_do * q.view(bs, t, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k_do * k.view(bs, t, 1, self.n_embd // self.n_head).transpose(1, 2)
        v = v_do * v.view(bs, t, 1, self.n_embd // self.n_head).transpose(1, 2)

        y = self.attn(q, k, v, mask)
        y = y.transpose(1, 2).contiguous().view(bs, t, self.n_embd)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MultiHeadAttention(SelfAttention):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__(config)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        device = x.device
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        ones = torch.ones((B, 1, T, 1), device=device)
        k_do = self.attn_dropout(ones)
        q_do = self.attn_dropout(ones)
        v_do = self.attn_dropout(ones)

        k = k_do * k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q_do * q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v_do * v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = self.attn(q, k, v, mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# -------------------------
# Feedforward blocks used by TransformerBlock
# -------------------------
class _MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, hidden_mult: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, int(hidden_mult * n_embd), bias=bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(int(hidden_mult * n_embd), n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class _MoEMLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, moe_config: Dict[str, Any]):
        super().__init__()
        # moe_config expected keys: ff_mult_factor, proj_features, num_experts, top_k, gate_sizes
        self.c_fc = MoELinear(
            n_embd,
            int(moe_config['ff_mult_factor'] * n_embd),
            proj_features=moe_config['proj_features'],
            num_experts=moe_config['num_experts'],
            bias=bias,
            top_k=moe_config.get('top_k', None),
            gate_sizes=tuple(moe_config.get('gate_sizes', [])),
        )
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = MoELinear(
            int(moe_config['ff_mult_factor'] * n_embd),
            n_embd,
            proj_features=moe_config['proj_features'],
            num_experts=moe_config['num_experts'],
            bias=bias,
            top_k=moe_config.get('top_k', None),
            gate_sizes=tuple(moe_config.get('gate_sizes', [])),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -------------------------
# TransformerBlock
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: Any, seed: Optional[int] = None, n_cls: int = 0):
        """
        config should expose:
         - is_causal (bool)
         - attn_config (SelfAttentionConfig-like): n_embd, n_head, attn_dropout, dropout, bias, pos_bias
         - rotator_config: either dict for MoE or numeric hidden mult for MLP path
         - is_sparse_attn (bool), max_block_size (optional), sparsity_factor (optional)
         - enable_gradient_checkpointing (bool)
        """
        super().__init__()
        self.is_causal = config.is_causal
        attn_cfg = config.attn_config
        self.ln_1 = LayerNorm(attn_cfg.n_embd, bias=attn_cfg.bias)
        self.attn = SelfAttention.from_config(attn_cfg)
        self.ln_2 = LayerNorm(attn_cfg.n_embd, bias=attn_cfg.bias)

        # rotator config can be either a dict (MoE) or a float/int for feedforward multiplier
        if isinstance(config.rotator_config, dict) and 'moe' in config.rotator_config:
            moe_cfg = config.rotator_config['moe']
            self.mlp = _MoEMLP(attn_cfg.n_embd, attn_cfg.bias, attn_cfg.dropout, moe_cfg)
        else:
            # expect numeric multiplier
            hidden_mult = config.rotator_config if isinstance(config.rotator_config, (int, float)) else 4
            self.mlp = _MLP(attn_cfg.n_embd, attn_cfg.bias, attn_cfg.dropout, hidden_mult)

        self.is_sparse = getattr(config, 'is_sparse_attn', False)
        self.enable_gradient_checkpointing = getattr(config, 'enable_gradient_checkpointing', False)

        if self.is_sparse:
            # For TorchScript safety, precompute deterministic permutation using torch.randperm with generator
            max_block_size = config.max_block_size
            sparsity_factor = config.sparsity_factor
            n_non_zeros = int(sparsity_factor * max_block_size)
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(seed)
            perm = torch.randperm(max_block_size, generator=g)
            full_mask = torch.cat((torch.arange(0, n_cls, dtype=torch.long),
                                   perm[n_cls:]), dim=0)
            # register buffers for indices
            idx_sorted, _ = full_mask[:n_non_zeros].sort()
            not_idx_sorted, _ = full_mask[n_non_zeros:].sort()
            self.register_buffer('input_mask_idx', idx_sorted, persistent=True)
            self.register_buffer('input_mask_not_idx', not_idx_sorted, persistent=True)
            self.null_connector = nn.Linear(attn_cfg.n_embd, attn_cfg.n_embd, bias=attn_cfg.bias)
        else:
            self.null_connector = nn.Identity()
            self.register_buffer('input_mask_idx', torch.empty(0, dtype=torch.long), persistent=True)
            self.register_buffer('input_mask_not_idx', torch.empty(0, dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_gradient_checkpointing and self.training:
            return self.checkpointed_forward(x, attn_mask=attn_mask)
        return self.inner_forward(x, attn_mask=attn_mask)

    def checkpointed_forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.utils.checkpoint.checkpoint(lambda *args: self.inner_forward(args[0], attn_mask=args[1]), x, attn_mask)

    def inner_forward(self, x_orig: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_sparse:
            T = x_orig.size(1)
            idx = self.input_mask_idx[self.input_mask_idx < T]
            if idx.numel() <= 1:
                return x_orig + self.null_connector(x_orig)
            not_idx = self.input_mask_not_idx[self.input_mask_not_idx < T]
            x = x_orig[:, idx]
            if attn_mask is not None:
                attn_mask = attn_mask[:, :, idx, :][:, :, :, idx]
        else:
            x = x_orig
            idx = torch.empty(0, dtype=torch.long, device=x_orig.device)
            not_idx = torch.empty(0, dtype=torch.long, device=x_orig.device)

        if self.is_causal:
            L = x.size(-2)
            device = x.device
            attn_mask_causal = torch.ones((L, L), device=device, dtype=torch.bool).tril(diagonal=0)
            attn_mask_causal = attn_mask_causal.float().masked_fill(~attn_mask_causal, -float('inf'))
            attn_mask_causal = attn_mask_causal.unsqueeze(0).unsqueeze(1)
        else:
            attn_mask_causal = None

        if attn_mask_causal is not None:
            if attn_mask is None:
                attn_mask = attn_mask_causal
            else:
                attn_mask = attn_mask + attn_mask_causal

        x = x + self.attn(self.ln_1(x), mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        if not self.is_sparse:
            return x

        x_final = torch.zeros_like(x_orig)
        x_final[:, idx] = x
        x_final[:, not_idx] = x_orig[:, not_idx] + self.null_connector(x_orig[:, not_idx])
        return x_final


# -------------------------
# SimhashVectorIndexer (bit-safe)
# -------------------------
class SimhashVectorIndexer(nn.Module):
    def __init__(self, inp_dim: int, n_proj: int = 16):
        super().__init__()
        self.register_buffer('projection_mat', torch.randn((inp_dim, n_proj)) / math.sqrt(float(inp_dim)), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., inp_dim)
        z = (x @ self.projection_mat) > 0  # bool
        res = torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device)
        for i in range(z.size(-1)):
            res = res + (z[..., i].long() << i)
        return res


# -------------------------
# CosineVectorEmbedding (bucketize + EmbeddingBag)
# -------------------------
class CosineVectorEmbedding(nn.Module):
    def __init__(self, inp_dim: int, emb_dim: int, n_proj: int = 16, num_bins: int = 20):
        super().__init__()
        proj = torch.randn((inp_dim, n_proj))
        proj = F.normalize(proj, p=2.0, dim=0)
        self.register_buffer('projection_mat', proj, persistent=True)

        resolution = 2.0 / float(num_bins)
        grid = torch.linspace(-1.0, 1.0, steps=num_bins + 1)[:-1] + 0.5 * resolution
        self.register_buffer('grid', grid, persistent=True)

        pos_offset = ((num_bins + 1) * torch.arange(0, n_proj, dtype=torch.long)).reshape(n_proj)
        self.register_buffer('pos_offset', pos_offset, persistent=True)

        self.emb = nn.EmbeddingBag((num_bins + 1) * n_proj, emb_dim, mode='sum')
        self.emb_dim = emb_dim
        self.n_proj = n_proj
        self.num_bins = num_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        z = F.normalize(x, p=2.0, dim=-1) @ self.projection_mat  # (bs, seq_len, n_proj)
        z_buckets = torch.bucketize(z, self.grid)  # (bs, seq_len, n_proj)
        z_flat = z_buckets.contiguous().view(-1, self.n_proj)  # (bs*seq_len, n_proj)
        offsets = self.pos_offset.unsqueeze(0)  # (1, n_proj)
        idxs = z_flat + offsets  # (bs*seq_len, n_proj)
        emb_out = self.emb(idxs)  # (bs*seq_len, emb_dim)
        emb_out = emb_out.view(bs, seq_len, self.emb_dim)
        return emb_out


# -------------------------
# QuantileMapper & DenseMapper
# -------------------------
class QuantileMapper(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        q = torch.tensor(quantiles)
        self.register_buffer('quantiles', q, persistent=True)
        self.n_bins = q.numel() + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, features?) likely (bs, 1) per feature
        bins = torch.bucketize(x, self.quantiles)  # same shape as x
        return bins.to(dtype=torch.float32) / float(self.n_bins) - 0.5


class DenseMapper(nn.Module):
    def __init__(self, stats: Dict[str, Any], emb_dim: int, n_projs: List[int], num_bins: List[int]):
        super().__init__()
        self.mappers = nn.ModuleDict({feature: QuantileMapper(stats[feature]) for feature in stats})
        assert len(n_projs) == len(num_bins)
        emb_list: List[nn.Module] = []
        for nproj, nbin in zip(n_projs, num_bins):
            emb_list.append(CosineVectorEmbedding(len(self.mappers), emb_dim, n_proj=nproj, num_bins=nbin))
        self.emb = nn.ModuleList(emb_list)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for feature, mod in self.mappers.items():
            parts.append(mod(batch[feature]))
        x = torch.cat(parts, dim=1).unsqueeze(1)
        out = None
        for i, emb in enumerate(self.emb):
            if out is None:
                out = emb(x)
            else:
                out = out + emb(x)
        return out.squeeze(1)


# -------------------------
# CosineLinear
# -------------------------
class CosineLinear(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        w = torch.randn((out_dim, inp_dim)) / math.sqrt(float(inp_dim))
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2.0, dim=-1), F.normalize(self.weight, p=2.0, dim=-1))


# -------------------------
# LearnableCosineVectorEmbedding (simplified)
# -------------------------
class LearnableCosineVectorEmbedding(nn.Module):
    def __init__(self,
                 inp_dim: int,
                 emb_dim: int,
                 n_proj: int = 16,
                 num_bins: int = 20,
                 sigma_inflation_factor: float = 1.0,
                 top_k: Optional[int] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_proj = n_proj
        self.num_bins = num_bins
        self.top_k = None if top_k is None else min(top_k, num_bins)
        self.sigma2 = (sigma_inflation_factor * 2.0 / num_bins) ** 2
        self.proj = CosineLinear(inp_dim, n_proj)
        # mean: (1, 1, n_proj, num_bins)
        mean = 2 * torch.rand((1, 1, n_proj, num_bins)) - 1
        self.mean = nn.Parameter(mean)
        self.emb = nn.Linear(self.n_proj * self.num_bins, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        z = self.gaussian_kernel(self.proj(x))  # (bs, seq_len, n_proj, num_bins)
        return self.emb(z.view(bs, seq_len, self.n_proj * self.num_bins))

    def gaussian_kernel(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - self.mean  # (bs, seq_len, n_proj, num_bins)
        act = torch.exp(-0.5 * diff * diff / float(self.sigma2))
        out = act.clone()
        if self.top_k is not None:
            top_k_vals, _ = torch.topk(act, k=self.top_k, dim=-1, largest=True, sorted=True)
            thresh = top_k_vals[..., -1].unsqueeze(-1)
            out = torch.where(act < thresh, torch.zeros_like(act), act)
        return F.normalize(out, p=2.0, dim=-1)


# -------------------------
# ProbabilityVectorEmbedding
# -------------------------
class ProbabilityVectorEmbedding(nn.Module):
    def __init__(self, emb_dim: int, num_bins: int = 10, sigma_inflation_factor: float = 1.0, top_k: Optional[int] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_bins = num_bins
        self.top_k = None if top_k is None else min(top_k, num_bins)
        self.sigma2 = (sigma_inflation_factor * 1.0 / num_bins) ** 2
        mean = torch.rand((1, 1, num_bins))
        self.mean = nn.Parameter(mean)
        self.emb = nn.Linear(self.num_bins, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, d = x.shape
        if d != 1:
            raise RuntimeError("ProbabilityVectorEmbedding expects input dim 1")
        z = self.gaussian_kernel(x)
        return self.emb(z.view(bs, -1))

    def gaussian_kernel(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - self.mean
        act = torch.exp(-0.5 * diff * diff / float(self.sigma2))
        out = act.clone()
        if self.top_k is not None:
            top_k_vals, _ = torch.topk(act, k=self.top_k, dim=-1, largest=True, sorted=True)
            thresh = top_k_vals[..., -1].unsqueeze(-1)
            out = torch.where(act < thresh, torch.zeros_like(act), act)
        return F.normalize(out, p=2.0, dim=-1)

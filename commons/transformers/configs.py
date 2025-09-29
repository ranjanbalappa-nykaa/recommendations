from typing import Optional, Union, Tuple
from pydantic import BaseModel
from enum import Enum


class MLPConfig(BaseModel):
    ff_mult: float


class MoEConfig(BaseModel):
    num_experts: int
    proj_features: int
    ff_mult_factor: float
    gate_sizes: Optional[Tuple[int, ...]] = None
    top_k: Optional[int] = None


class SelfAttentionType(Enum):
    MULTI_HEAD: str = 'multi_head'
    MULTI_QUERY: str = 'multi_query'


class PositionBiasConfig(BaseModel):
    context_window: int


class SelfAttentionConfig(BaseModel):
    attn_dropout: float = 0.1
    bias: bool = True
    dropout: float = 0.1
    n_head: int = 12
    n_embd: int = 768
    pos_bias: Optional[PositionBiasConfig] = None
    attn_type: SelfAttentionType


class TransformerConfig(BaseModel):
    rotator_config: Union[MoEConfig, MLPConfig]
    is_causal: bool = False
    max_block_size: Optional[int] = None
    is_sparse_attn: bool = False
    sparsity_factor: float = 0.5
    enable_gradient_checkpointing: bool = False
    attn_config: SelfAttentionConfig

"""
Decoder Layer 实现
整合 Attention 和 MoE，支持 PreNorm 结构
"""
import torch
import torch.nn as nn
from typing import Optional

from cache_utils import SimpleKVCache
from attention import LlamaLikeAttention, create_causal_mask
from moe import StochasticMoELayer


class LlamaLikeRMSNorm(nn.Module):
    """
    RMS Normalization
    参考 T5LayerNorm 实现
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [..., hidden_size]
        
        Returns:
            normalized: [..., hidden_size]
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaLikeDecoderLayer(nn.Module):
    """
    Llama-like Decoder Layer
    结构:
        residual = x
        x = input_layernorm(x)
        x = self_attn(x, past_key_values, cache_position)
        x = residual + x
        
        residual = x
        x = post_attention_layernorm(x)
        x = moe(x)
        x = residual + x
    """
    
    def __init__(self, config: dict, layer_idx: int, use_moe: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.get('hidden_size', 4096)
        self.use_moe = use_moe
        
        # Attention
        self.self_attn = LlamaLikeAttention(config, layer_idx)
        
        # MoE 或 MLP
        if use_moe:
            self.mlp = StochasticMoELayer(config, layer_idx)
        else:
            # 简单的 SwiGLU MLP
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size, bias=False),
            )
        
        # Layer Norm
        self.input_layernorm = LlamaLikeRMSNorm(self.hidden_size, eps=config.get('rms_norm_eps', 1e-6))
        self.post_attention_layernorm = LlamaLikeRMSNorm(self.hidden_size, eps=config.get('rms_norm_eps', 1e-6))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor = None,
        past_key_values: Optional[SimpleKVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置 ID [batch, seq_len]
            past_key_values: KV Cache
            cache_position: Cache 位置
            use_router_logits: 是否使用路由器 logits
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # 1. Self Attention 分支
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        
        # 残差连接
        hidden_states = residual + hidden_states
        
        # 2. MLP/MoE 分支
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP/MoE
        if self.use_moe:
            hidden_states = self.mlp(hidden_states, use_router_logits=use_router_logits)
        else:
            hidden_states = self.mlp(hidden_states)
        
        # 残差连接
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    def get_gpu_cache_size(self) -> int:
        """获取 MoE GPU 缓存大小"""
        if self.use_moe:
            return self.mlp.get_gpu_cache_size()
        return 0

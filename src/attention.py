"""
Attention 模块实现
参考 Qwen3 的设计，支持 RoPE、KV Cache、GQA
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

from cache_utils import SimpleKVCache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将张量的一半维度旋转 -90 度
    用于 RoPE 计算
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码 (RoPE)
    
    Args:
        q: query tensor [batch, heads, seq_len, head_dim]
        k: key tensor [batch, heads, seq_len, head_dim]
        cos: cosine embedding [batch, seq_len, head_dim]
        sin: sine embedding [batch, seq_len, head_dim]
        position_ids: 位置 ID（已弃用）
        unsqueeze_dim: unsqueeze 的维度
    
    Returns:
        (rotated_q, rotated_k)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 K/V 以支持 GQA (Grouped Query Attention)
    
    将 K/V 从 [batch, num_kv_heads, seq_len, head_dim]
    重复到 [batch, num_heads, seq_len, head_dim]
    
    Args:
        hidden_states: K/V 张量
        n_rep: 重复次数 = num_heads / num_kv_heads
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    """
    
    inv_freq: torch.Tensor
    
    def __init__(self, config: dict, device=None):
        super().__init__()
        self.max_seq_len_cached = config.get('max_position_embeddings', 4096)
        self.rope_theta = config.get('rope_theta', 10000.0)
        
        dim = config.get('head_dim', config.get('hidden_size', 4096) // config.get('num_attention_heads', 32))
        
        # 计算逆频率
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 cos 和 sin 嵌入
        
        Args:
            x: 输入张量 [batch, seq_len, ...]
            position_ids: 位置 ID [batch, seq_len]
        
        Returns:
            (cos, sin) embeddings [batch, seq_len, head_dim]
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # 计算频率
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLikeAttention(nn.Module):
    """
    Llama-like Attention 模块
    支持:
    - KV Cache (prefill/decode)
    - RoPE (旋转位置编码)
    - GQA (Grouped Query Attention)
    """
    
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 模型配置
        self.hidden_size = config.get('hidden_size', 4096)
        self.num_heads = config.get('num_attention_heads', 32)
        self.num_kv_heads = config.get('num_key_value_heads', 32)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = config.get('max_position_embeddings', 4096)
        
        # 投影层
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(config)
        
        # 缩放因子
        self.scaling = self.head_dim ** (-0.5)
        
        # 是否为因果注意力
        self.is_causal = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[SimpleKVCache],
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_values: KV Cache
            attention_mask: 注意力掩码
            position_ids: 位置 ID [batch, seq_len]
            cache_position: cache 位置 [seq_len]
        
        Returns:
            attn_output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # 1. 投影到 Q/K/V
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # 2. 应用 RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 3. 更新 KV Cache (如果启用)
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        # 4. 重复 K/V 以支持 GQA（标准 Attention 需要）
        # Flash Attention 内部会处理 GQA，不需要预先重复
        # Debug: 打印 shape
        print(f"[Attention] Before repeat_kv: Q={query_states.shape}, K={key_states.shape}, V={value_states.shape}")
        print(f"[Attention] num_key_value_groups={self.num_key_value_groups}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")
        print(f"[Attention] head_dim Q={query_states.shape[-1]}, K={key_states.shape[-1]}, V={value_states.shape[-1]}")
        
        # 确保 head_dim 一致
        if key_states.shape[-1] != query_states.shape[-1]:
            print(f"[Attention] WARNING: head_dim mismatch! Resizing K/V from {key_states.shape[-1]} to {query_states.shape[-1]}")
            key_states = key_states[..., :query_states.shape[-1]]
            value_states = value_states[..., :query_states.shape[-1]]
        
        # 5. 计算注意力（Flash Attention 接收原始的 K/V）
        attn_output = self._compute_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        # 6. 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        计算注意力 (使用 Flash Attention 或标准实现)
        
        Args:
            query_states: [batch, num_heads, seq_len, head_dim]
            key_states: [batch, num_kv_heads, kv_seq_len, head_dim]
            value_states: [batch, num_kv_heads, kv_seq_len, head_dim]
            attention_mask: 注意力掩码
        
        Returns:
            attn_output: [batch, seq_len, num_heads * head_dim]
        """
        # 尝试使用 Flash Attention 2（Flash Attention 内部处理 GQA）
        try:
            from flash_attn import flash_attn_func
            print(f"[Attention] Using Flash Attention with original K/V shapes: K={key_states.shape}, V={value_states.shape}")
            
            # Flash Attention 2 期望: [batch, seq_len, num_heads, head_dim]
            # 我们当前是: [batch, num_heads, seq_len, head_dim]
            # 需要转置
            q_fa = query_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
            k_fa = key_states.transpose(1, 2)    # [batch, num_kv_heads, kv_seq_len, head_dim] -> [batch, kv_seq_len, num_kv_heads, head_dim]
            v_fa = value_states.transpose(1, 2)  # [batch, num_kv_heads, kv_seq_len, head_dim] -> [batch, kv_seq_len, num_kv_heads, head_dim]
            
            print(f"[Attention] Flash Attention input shapes: Q={q_fa.shape}, K={k_fa.shape}, V={v_fa.shape}")
            
            attn_output = flash_attn_func(
                q_fa,
                k_fa,
                v_fa,
                causal=self.is_causal,
            )
            
            # Flash Attention 输出: [batch, seq_len, num_heads, head_dim]
            # 转回标准形状: [batch, num_heads, seq_len, head_dim]
            attn_output = attn_output.transpose(1, 2)
            
        except ImportError:
            # 退回到标准实现（需要手动重复 K/V）
            key_states_repeated = repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeated = repeat_kv(value_states, self.num_key_value_groups)
            
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) * self.scaling
            
            if attention_mask is not None:
                # 应用因果掩码
                attn_weights = attn_weights + attention_mask
            
            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # 应用到 value
            attn_output = torch.matmul(attn_weights, value_states_repeated)
        
        # 转换形状: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Reshape: [batch, seq_len, num_heads * head_dim]
        batch_size, seq_length, num_heads, head_dim = attn_output.shape
        attn_output = attn_output.reshape(batch_size, seq_length, num_heads * head_dim)
        
        return attn_output


def create_causal_mask(
    config: dict,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.LongTensor],
    past_key_values: Optional[SimpleKVCache],
    position_ids: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    创建因果掩码
    
    Args:
        config: 配置字典
        input_embeds: 输入 embeddings
        attention_mask: 原始注意力掩码
        cache_position: cache 位置
        past_key_values: KV Cache
        position_ids: 位置 ID
    
    Returns:
        causal_mask: 因果掩码张量
    """
    # 如果使用 Flash Attention，可以返回 None
    # 这里实现标准因果掩码
    try:
        from flash_attn import flash_attn_func
        return None
    except ImportError:
        batch_size, seq_length = input_embeds.shape[:2]
        
        # 获取 KV 序列长度
        kv_seq_length = seq_length
        if past_key_values is not None:
            kv_seq_length = past_key_values.get_seq_length()
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones((seq_length, kv_seq_length), dtype=torch.bool),
            diagonal=kv_seq_length - seq_length + 1,
        )
        
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        causal_mask = causal_mask.float()
        causal_mask.masked_fill_(causal_mask.bool(), float('-inf'))
        
        return causal_mask.to(input_embeds.device)

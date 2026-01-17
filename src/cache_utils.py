"""
KV Cache 实现
参考 Qwen3/DeepSeek 的 DynamicCache 设计
"""
import torch
from typing import Optional, Tuple


class SimpleKVCache:
    """
    简化的 KV Cache 实现
    支持按层存储 key/value，用于 prefill 和 decode 阶段
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，包含:
                - num_hidden_layers: 层数
                - num_attention_heads: 注意力头数
                - num_key_value_heads: KV 头数 (用于 GQA)
                - hidden_size: 隐藏层大小
                - head_dim: 每个 head 的维度
        """
        self.config = config
        self.num_layers = config.get('num_hidden_layers', 10)
        self.num_heads = config.get('num_attention_heads', 32)
        self.num_kv_heads = config.get('num_key_value_heads', 32)
        self.head_dim = config.get('head_dim', config.get('hidden_size', 4096) // 32)
        
        # 按层存储 key/value，每层是一个 list of tensors
        self.key_cache: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self.value_cache: list[list[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        
        # 跟踪每层的序列长度
        self.seq_length_per_layer: list[int] = [0] * self.num_layers
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新指定层的 KV cache
        
        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: 包含 cache_position 等信息的字典
        
        Returns:
            拼接后的完整 key_states 和 value_states
        """
        if cache_kwargs is None:
            cache_kwargs = {}
        
        # 获取当前的 cache position
        cache_position = cache_kwargs.get('cache_position', None)
        
        # 添加到缓存
        self.key_cache[layer_idx].append(key_states)
        self.value_cache[layer_idx].append(value_states)
        
        # 保存原始 dtype
        original_dtype = key_states.dtype
        
        # 更新序列长度
        batch_size = key_states.shape[0]
        if cache_position is not None:
            self.seq_length_per_layer[layer_idx] = max(
                self.seq_length_per_layer[layer_idx],
                cache_position[-1].item() + 1 if hasattr(cache_position[-1], 'item') else cache_position[-1] + 1
            )
        else:
            self.seq_length_per_layer[layer_idx] += key_states.shape[2]
        
        # 拼接所有缓存的 key/value（保持原始 dtype）
        if len(self.key_cache[layer_idx]) > 0:
            all_keys = torch.cat(self.key_cache[layer_idx], dim=2).to(original_dtype)
            all_values = torch.cat(self.value_cache[layer_idx], dim=2).to(original_dtype)
        else:
            all_keys = key_states
            all_values = value_states
        
        return all_keys, all_values
    
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """
        获取序列长度
        
        Args:
            layer_idx: 层索引，如果为 None 则返回所有层的最大序列长度
        """
        if layer_idx is not None:
            return self.seq_length_per_layer[layer_idx]
        return max(self.seq_length_per_layer) if self.seq_length_per_layer else 0
    
    def get(
        self,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取指定层的所有缓存 key/value
        
        Returns:
            (key_states, value_states) 或 (None, None) 如果缓存为空
        """
        if not self.key_cache[layer_idx]:
            return None, None
        
        all_keys = torch.cat(self.key_cache[layer_idx], dim=2)
        all_values = torch.cat(self.value_cache[layer_idx], dim=2)
        return all_keys, all_values
    
    def reset(self):
        """清空缓存（用于新对话）"""
        self.key_cache = [[] for _ in range(self.num_layers)]
        self.value_cache = [[] for _ in range(self.num_layers)]
        self.seq_length_per_layer = [0] * self.num_layers


class DynamicCache(SimpleKVCache):
    """
    扩展的 Dynamic Cache，支持动态调整
    为了兼容 transformers 的接口命名
    """
    pass

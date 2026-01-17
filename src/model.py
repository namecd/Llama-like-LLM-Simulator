"""
主模型实现和评测脚本
支持 Prefill 和 Decode 两种模式
"""
import random
import time
import torch
import torch.nn as nn
from typing import Optional, Tuple

from cache_utils import SimpleKVCache
from layer import LlamaLikeDecoderLayer, LlamaLikeRMSNorm


class LlamaLikeMoEModel(nn.Module):
    """
    Llama-like MoE 模型
    支持 Prefill 和 Decode 两种模式
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 模型配置
        self.vocab_size = config.get('vocab_size', 32000)
        self.hidden_size = config.get('hidden_size', 4096)
        self.num_hidden_layers = config.get('num_hidden_layers', 10)
        self.max_position_embeddings = config.get('max_position_embeddings', 4096)
        self.rms_norm_eps = config.get('rms_norm_eps', 1e-6)
        
        # Embedding 层（用于模拟输入）
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            LlamaLikeDecoderLayer(config, layer_idx, use_moe=config.get('use_moe', True))
            for layer_idx in range(self.num_hidden_layers)
        ])
        
        # Final Norm
        self.norm = LlamaLikeRMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        
        # KV Cache
        self.past_key_values: Optional[SimpleKVCache] = None
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[SimpleKVCache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        use_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SimpleKVCache]]:
        """
        前向传播
        
        Args:
            input_ids: [batch, seq_len] 输入 token IDs
            inputs_embeds: [batch, seq_len, hidden_size] 输入 embeddings（优先级高于 input_ids）
            attention_mask: 注意力掩码
            position_ids: [batch, seq_len] 位置 ID
            past_key_values: KV Cache
            use_cache: 是否使用 cache
            cache_position: Cache 位置
            use_router_logits: 是否使用路由器 logits
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_values: 更新后的 KV Cache
        """
        # 1. 准备输入 embeddings
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.embed_tokens(input_ids)
                # 转换为与模型相同的精度
                model_dtype = self.layers[0].self_attn.q_proj.weight.dtype
                if inputs_embeds.dtype != model_dtype:
                    inputs_embeds = inputs_embeds.to(model_dtype)
            else:
                raise ValueError("Must provide either input_ids or inputs_embeds")
        
        batch_size, seq_length, _ = inputs_embeds.shape
        
        # 2. 初始化 KV Cache
        if use_cache and past_key_values is None:
            past_key_values = SimpleKVCache(self.config)
        
        # 3. 准备 position_ids
        if position_ids is None:
            past_seen_tokens = 0
            if past_key_values is not None:
                past_seen_tokens = past_key_values.get_seq_length()
            
            if cache_position is None:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
                )
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        elif cache_position is None:
            cache_position = torch.arange(seq_length, device=inputs_embeds.device)
        
        # 4. 创建因果掩码（标准 Attention 需要，Flash Attention 可以跳过）
        causal_mask = None
        # 注意：Flash Attention 会自动处理因果掩码，不需要手动创建
        
        # 5. 通过所有层
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_router_logits=use_router_logits,
            )
        
        # 6. Final Norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, past_key_values
    
    def reset_cache(self):
        """重置 KV Cache（用于新对话）"""
        if self.past_key_values is not None:
            self.past_key_values.reset()


def run_prefill_simulation(
    model: LlamaLikeMoEModel,
    config: dict,
    num_layers: int = 10,
    seq_len: int = 256,
    use_profiler: bool = False,
):
    """
    运行 Prefill 阶段模拟
    
    Args:
        model: 模型
        config: 配置
        num_layers: 模拟的层数
        seq_len: 序列长度
        use_profiler: 是否使用 profiler
    """
    print(f"\n{'='*60}")
    print(f"Prefill Simulation (Sequence Length: {seq_len})")
    print(f"{'='*60}")
    
    # 1. 创建输入（模拟）
    batch_size = 1
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device='cuda')
    
    # 2. 预热
    print("预热中...")
    for _ in range(2):
        _, _ = model(
            input_ids=input_ids,
            use_cache=True,
            use_router_logits=config.get('use_router_logits', False),
        )
    torch.cuda.synchronize()
    
    print("预热结束，开始测试...\n")
    
    # 3. 注册 profiler hooks
    if use_profiler:
        try:
            from NsightProfiler import NsightProfiler
            NsightProfiler.register_layer_hooks(model)
            NsightProfiler.wrap_method(model, 'self_attn', tag_name='1_Attention_Phase')
            NsightProfiler.wrap_method(model.layers[0].mlp, 'forward', tag_name='2_MoE_Phase')
        except ImportError:
            print("Warning: NsightProfiler not available")
    
    # 4. 运行测试
    start_time = time.time()
    
    # 只使用前 num_layers 层
    for layer_idx in range(min(num_layers, len(model.layers))):
        hidden_states = torch.randn(
            batch_size, seq_len, config['hidden_size'],
            device='cuda', dtype=torch.float16
        )
        
        # 直接调用单层（模拟层间串行）
        layer = model.layers[layer_idx]
        attention_mask = None
        position_ids = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(seq_len, device='cuda')
        
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,  # Prefill 每层重新计算
            cache_position=cache_position,
            use_router_logits=config.get('use_router_logits', False),
        )
        
        print(f"Layer {layer_idx + 1}/{num_layers} 完成 | GPU Cache: {layer.get_gpu_cache_size()} experts")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Prefill 总耗时: {end_time - start_time:.4f} s")
    print(f"平均每层耗时: {(end_time - start_time) / num_layers:.4f} s")
    print(f"{'='*60}\n")
    
    return end_time - start_time


def run_decode_simulation(
    model: LlamaLikeMoEModel,
    config: dict,
    num_layers: int = 10,
    prefill_seq_len: int = 256,
    decode_steps: int = 10,
    use_profiler: bool = False,
):
    """
    运行 Decode 阶段模拟
    
    Args:
        model: 模型
        config: 配置
        num_layers: 模拟的层数
        prefill_seq_len: Prefill 序列长度
        decode_steps: Decode 步数
        use_profiler: 是否使用 profiler
    """
    print(f"\n{'='*60}")
    print(f"Decode Simulation (Prefill: {prefill_seq_len}, Decode Steps: {decode_steps})")
    print(f"{'='*60}")
    
    batch_size = 1
    
    # 1. Prefill 阶段：填充 KV Cache
    print(f"Prefill 阶段：填充 {prefill_seq_len} tokens...")
    prefill_input_ids = torch.randint(0, config['vocab_size'], (batch_size, prefill_seq_len), device='cuda')
    
    past_key_values = SimpleKVCache(config)
    
    # 运行所有层的 prefill
    for layer_idx in range(min(num_layers, len(model.layers))):
        hidden_states = torch.randn(
            batch_size, prefill_seq_len, config['hidden_size'],
            device='cuda', dtype=torch.float16
        )
        
        layer = model.layers[layer_idx]
        attention_mask = None
        position_ids = torch.arange(prefill_seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(prefill_seq_len, device='cuda')
        
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_router_logits=config.get('use_router_logits', False),
        )
    
    print(f"Prefill 完成，KV Cache 长度: {past_key_values.get_seq_length()}\n")
    
    # 2. Decode 阶段：逐步生成
    print(f"Decode 阶段：生成 {decode_steps} tokens...")
    
    # 注册 profiler hooks
    if use_profiler:
        try:
            from NsightProfiler import NsightProfiler
            NsightProfiler.register_layer_hooks(model)
            NsightProfiler.wrap_method(model, 'self_attn', tag_name='1_Attention_Phase')
            NsightProfiler.wrap_method(model.layers[0].mlp, 'forward', tag_name='2_MoE_Phase')
        except ImportError:
            print("Warning: NsightProfiler not available")
    
    decode_start = time.time()
    for step in range(decode_steps):
        step_start = time.time()
        
        # 生成单个 token
        for layer_idx in range(min(num_layers, len(model.layers))):
            hidden_states = torch.randn(
                batch_size, 1, config['hidden_size'],
                device='cuda', dtype=torch.float16
            )
            
            layer = model.layers[layer_idx]
            attention_mask = None
            position_ids = torch.tensor([[prefill_seq_len + step]], device='cuda')
            cache_position = torch.tensor([prefill_seq_len + step], device='cuda')
            
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_router_logits=config.get('use_router_logits', False),
            )
        
        torch.cuda.synchronize()
        step_end = time.time()
        
        print(f"Step {step + 1}/{decode_steps} | 耗时: {step_end - step_start:.4f} s | KV Cache: {past_key_values.get_seq_length()} tokens")
    
    decode_end = time.time()
    
    print(f"\n{'='*60}")
    print(f"Decode 总耗时: {decode_end - decode_start:.4f} s")
    print(f"平均每 token 耗时: {(decode_end - decode_start) / decode_steps:.4f} s")
    print(f"Tokens/s: {decode_steps / (decode_end - decode_start):.2f}")
    print(f"{'='*60}\n")
    
    return decode_end - decode_start


def main():
    """主函数"""
    # 配置参数
    config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 10,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,  # 可改为 8/16 支持 GQA
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "head_dim": 128,  # hidden_size / num_attention_heads
        "rope_theta": 10000.0,
        
        # MoE 配置
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "intermediate_size": 16384,  # 4 * hidden_size
        "norm_topk_prob": True,
        "use_router_logits": False,  # 是否使用路由器 logits（模拟开销）
        
        # 模拟配置
        "gpu_capacity": 8,
        "prefetch_num": 4,
        "prefetch_acc": 0.8,
        "use_moe": True,
    }
    
    print(f"{'='*60}")
    print("Llama-like MoE LLM Simulator")
    print(f"{'='*60}")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Num Layers: {config['num_hidden_layers']}")
    print(f"Num Experts: {config['num_experts']}")
    print(f"Active Experts per Token: {config['num_experts_per_tok']}")
    print(f"GPU Capacity: {config['gpu_capacity']}/{config['num_experts']}")
    print(f"Prefetch Accuracy: {config['prefetch_acc']}")
    print(f"{'='*60}\n")
    
    # 创建模型
    model = LlamaLikeMoEModel(config)
    model = model.cuda().half()
    model.eval()
    
    # 运行 Prefill 模拟
    prefill_time = run_prefill_simulation(
        model=model,
        config=config,
        num_layers=10,
        seq_len=256,
        use_profiler=True,
    )
    
    # 运行 Decode 模拟
    decode_time = run_decode_simulation(
        model=model,
        config=config,
        num_layers=10,
        prefill_seq_len=256,
        decode_steps=10,
        use_profiler=True,
    )
    
    # 总结
    print(f"\n{'='*60}")
    print("性能总结")
    print(f"{'='*60}")
    print(f"Prefill (256 tokens): {prefill_time:.4f} s")
    print(f"Decode (10 tokens): {decode_time:.4f} s")
    print(f"Decode Tokens/s: {10 / decode_time:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

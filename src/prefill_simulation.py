"""
    一个功能更完备、细节实现更多的模拟，现在已经不用了
    模拟的入口看easy_simulate.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional
from dataclasses import dataclass
from NsightProfiler import NsightProfiler


@dataclass
class SimulationConfig:
    """配置模拟参数"""
    num_attention_heads: int = 16
    hidden_size: int = 2048
    num_key_value_heads: int = 16  # KV头的数量
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_experts: int = 2
    seq_len: int = 512
    intermediate_size: int = 5120
    prediction_correctness: float = 0.8  # 预测准确率
    prefill_predict_num: int = 4  # 预测要使用的专家数量
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FlashAttention(nn.Module):
    """模拟Flash Attention的实现"""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        # 确保维度匹配
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = query.shape
        q_dim = self.num_attention_heads
        kv_dim = self.num_key_value_heads
        head_dim = self.head_dim
        
        # 重塑Q, K, V张量
        query = query.view(bsz, seq_len, q_dim, head_dim).transpose(1, 2)  # (bsz, q_dim, seq_len, head_dim)
        key = key.view(bsz, seq_len, kv_dim, head_dim).transpose(1, 2)    # (bsz, kv_dim, seq_len, head_dim)
        value = value.view(bsz, seq_len, kv_dim, head_dim).transpose(1, 2)  # (bsz, kv_dim, seq_len, head_dim)
        
        # 扩展KV头以匹配Q头的数量 (GQA - Grouped Query Attention)
        if self.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, repeats=self.num_key_value_groups, dim=2)
            value = torch.repeat_interleave(value, repeats=self.num_key_value_groups, dim=2)
        
        # 计算attention分数
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, value)  # (bsz, q_dim, seq_len, head_dim)
        
        # 重新排列维度
        attn_output = attn_output.transpose(1, 2).contiguous()  # (bsz, seq_len, q_dim, head_dim)
        attn_output = attn_output.view(bsz, seq_len, self.hidden_size)  # (bsz, seq_len, hidden_size)
        
        return attn_output


class Expert(nn.Module):
    """专家网络，基于DeepSeekV3实现"""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down


class ExpertGate(nn.Module):
    """专家门控机制"""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((config.num_experts, config.hidden_size)))  # (experts_num, hidden_dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # batch_size * seq_len, hidden_dim
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits  # (batch_size * seq_len, experts_num)


class MoELayer(nn.Module):
    """MoE层实现，参考DeepSeekV3MoE"""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = config.shared_experts

        # 创建专家
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        
        # 共享专家
        if self.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.n_shared_experts)])
        
        # 门控
        self.gate = ExpertGate(config)
        
        # 用于模拟GPU/CPU专家切换的集合
        self.gpu_experts = set()
        self.cpu_experts = set(range(config.num_experts))

    def _compute_router_mask(self, router_logits):
        """计算路由掩码和权重"""
        # 使用sigmoid而不是softmax来模拟路由决策
        router_scores = router_logits.sigmoid()
        
        # 获取top-k专家
        topk_weights, topk_indices = torch.topk(router_scores, self.top_k, dim=-1)
        
        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # 计算路由logits
        router_logits = self.gate(hidden_states)
        
        # 获取路由权重和索引
        topk_weights, topk_indices = self._compute_router_mask(router_logits)
        
        # (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # 初始化输出
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # 按专家分组处理tokens（更高效的实现）
        # 创建专家到tokens的映射
        expert_to_tokens = {}
        for token_idx in range(hidden_states.size(0)):
            for expert_idx in topk_indices[token_idx]:
                expert_idx = expert_idx.item()
                weight = topk_weights[token_idx, topk_indices[token_idx] == expert_idx].item()
                if expert_idx not in expert_to_tokens:
                    expert_to_tokens[expert_idx] = []
                expert_to_tokens[expert_idx].append((token_idx, weight))
        
        # 每个专家处理分配给它的tokens
        for expert_idx, token_weight_pairs in expert_to_tokens.items():
            # 获取分配给此专家的tokens
            token_indices = [pair[0] for pair in token_weight_pairs]
            weights = torch.tensor([pair[1] for pair in token_weight_pairs], 
                                   dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
            
            # 获取对应的输入
            expert_input = hidden_states[token_indices]
            
            # 通过专家处理
            expert_output = self.experts[expert_idx](expert_input)
            
            # 加权并累加到最终输出
            weighted_output = weights * expert_output
            final_hidden_states[token_indices] += weighted_output
        
        # 处理共享专家
        if self.n_shared_experts > 0:
            for shared_expert in self.shared_experts:
                shared_output = shared_expert(hidden_states)
                final_hidden_states += shared_output
        
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


class DecodingLayer(nn.Module):
    """解码层实现"""
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        
        # 注意力相关的线性层
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * (config.hidden_size // config.num_attention_heads), bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 注意力机制
        self.attn = FlashAttention(config)
        
        # MoE层
        self.moe = MoELayer(config)
        
        # LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor):
        # 自注意力
        residual = x
        x = self.input_layernorm(x)
        
        # 生成Q, K, V
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)
        
        # 执行attention
        attn_output = self.attn(query_states, key_states, value_states)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        # 残差连接
        x = residual + attn_output
        
        # MoE处理
        residual = x
        x = self.post_attention_layernorm(x)
        moe_output = self.moe(x)
        x = residual + moe_output
        
        # 模拟专家预取
        self._prefetch_experts(x)
        
        return x
    
    def _prefetch_experts(self, x: torch.Tensor):
        """模拟专家预取"""
        # 重新实现路由，因为我们需要所有专家的分数
        router_logits = F.linear(x[:, -1:, :].type(torch.float32), 
                                 self.moe.gate.weight.type(torch.float32))
        router_scores = router_logits.sigmoid()
        
        # 获取预测的专家
        _, top_experts = torch.topk(router_scores[0, 0, :], 
                                   min(self.config.prefill_predict_num, self.config.num_experts))
        
        # 模拟预取逻辑，根据预测准确率决定哪些专家会被真正用到
        for expert_idx in top_experts:
            if torch.rand(1) < self.config.prediction_correctness:
                # 预测正确的专家，移动到GPU
                if expert_idx.item() in self.moe.cpu_experts:
                    self.moe.cpu_experts.remove(expert_idx.item())
                    self.moe.gpu_experts.add(expert_idx.item())
            else:
                # 预测错误，这些专家会从GPU移回CPU（如果在GPU上）
                if expert_idx.item() in self.moe.gpu_experts:
                    self.moe.gpu_experts.remove(expert_idx.item())
                    self.moe.cpu_experts.add(expert_idx.item())


class PrefillSimulator:
    """Prefill过程模拟器"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        # 预先初始化所有权重矩阵
        self.layer = DecodingLayer(config).to(config.device)
        self._initialize_weights()
        
    def _initialize_weights(self):
        """统一初始化所有权重矩阵，避免在执行时分配"""
        with torch.no_grad():
            # 初始化DecodingLayer中的权重
            nn.init.xavier_uniform_(self.layer.q_proj.weight)
            nn.init.xavier_uniform_(self.layer.k_proj.weight)
            nn.init.xavier_uniform_(self.layer.v_proj.weight)
            nn.init.xavier_uniform_(self.layer.o_proj.weight)
            
            # 初始化MoE中的门控权重
            nn.init.xavier_uniform_(self.layer.moe.gate.weight)
            
            # 初始化所有专家的权重
            for expert in self.layer.moe.experts:
                nn.init.xavier_uniform_(expert.gate_proj.weight)
                nn.init.xavier_uniform_(expert.up_proj.weight)
                nn.init.xavier_uniform_(expert.down_proj.weight)
            
            # 初始化共享专家的权重
            if self.layer.moe.n_shared_experts > 0:
                for shared_expert in self.layer.moe.shared_experts:
                    nn.init.xavier_uniform_(shared_expert.gate_proj.weight)
                    nn.init.xavier_uniform_(shared_expert.up_proj.weight)
                    nn.init.xavier_uniform_(shared_expert.down_proj.weight)

    def simulate(self, seq_len: Optional[int] = None, 
                 num_experts: Optional[int] = None,
                 num_experts_per_tok: Optional[int] = None,
                 prediction_correctness: Optional[float] = None) -> dict:
        """执行模拟测试"""
        # 更新配置参数（如果提供了）
        if seq_len is not None:
            self.config.seq_len = seq_len
        if num_experts is not None:
            self.config.num_experts = num_experts
        if num_experts_per_tok is not None:
            self.config.num_experts_per_tok = num_experts_per_tok
        if prediction_correctness is not None:
            self.config.prediction_correctness = prediction_correctness
            
        # 重新初始化层以适应新参数
        self.layer = DecodingLayer(self.config).to(self.config.device)
        self._initialize_weights()

        # 劫持特定方法进行分析
        NsightProfiler.register_layer_hooks(self.layer)
        
        # 创建随机输入
        batch_size = 1
        input_tensor = torch.randn(batch_size, self.config.seq_len, self.config.hidden_size, 
                                  device=self.config.device)
        
        # 预热
        # for _ in range(3):
        #     _ = self.layer(input_tensor)
        
        # 计时测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        output = self.layer(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 返回性能指标
        return {
            "execution_time": execution_time,
            "seq_len": self.config.seq_len,
            "num_experts": self.config.num_experts,
            "num_experts_per_tok": self.config.num_experts_per_tok,
            "prediction_correctness": self.config.prediction_correctness,
            "memory_used": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "output_shape": tuple(output.shape)
        }


def run_simulation():
    """运行模拟测试"""
    config = SimulationConfig()
    simulator = PrefillSimulator(config)
    
    print("开始Prefill过程模拟测试...")
    print("="*50)
    
    # 测试不同的序列长度
    print("\n测试不同序列长度:")
    for seq_len in [128, 256, 512, 1024]:
        result = simulator.simulate(seq_len=seq_len)
        print(f"序列长度: {result['seq_len']}, 执行时间: {result['execution_time']:.4f}s, "
              f"内存使用: {result['memory_used']/1024/1024:.2f}MB")
    
    # 测试不同专家数量
    print("\n测试不同专家数量:")
    simulator.config.seq_len = 512  # 重置序列长度
    for num_experts in [4, 8, 16, 32]:
        result = simulator.simulate(num_experts=num_experts)
        print(f"专家数量: {result['num_experts']}, 执行时间: {result['execution_time']:.4f}s, "
              f"内存使用: {result['memory_used']/1024/1024:.2f}MB")
    
    # 测试不同激活专家数量
    print("\n测试不同激活专家数量:")
    simulator.config.num_experts = 8  # 重置专家数量
    for num_experts_per_tok in [1, 2, 4]:
        result = simulator.simulate(num_experts_per_tok=num_experts_per_tok)
        print(f"激活专家数: {result['num_experts_per_tok']}, 执行时间: {result['execution_time']:.4f}s, "
              f"内存使用: {result['memory_used']/1024/1024:.2f}MB")
    
    # 测试不同预测准确率
    print("\n测试不同预测准确率:")
    simulator.config.num_experts_per_tok = 2  # 重置激活专家数量
    for pred_correctness in [0.5, 0.7, 0.9]:
        result = simulator.simulate(prediction_correctness=pred_correctness)
        print(f"预测准确率: {result['prediction_correctness']}, 执行时间: {result['execution_time']:.4f}s, "
              f"内存使用: {result['memory_used']/1024/1024:.2f}MB")


if __name__ == "__main__":
    run_simulation()
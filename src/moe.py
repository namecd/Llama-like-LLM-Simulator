"""
MoE (Mixture of Experts) 模块实现
参考 Qwen3/DeepSeek 的设计，支持路由计算和专家执行
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LlamaLikeExpert(nn.Module):
    """
    标准的 SwiGLU 专家结构 (used in Llama, Mixtral, DeepSeek, etc.)
    包含 3 个线性层：Gate, Up, Down
    """
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        # 默认 intermediate_size 通常是 hidden_size 的 4 倍
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        
        Args:
            x: [batch, seq_len, hidden_size] 或 [num_tokens, hidden_size]
        
        Returns:
            output: 相同形状
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):
    """
    Top-K 路由器
    计算每个 token 应该路由到哪些专家
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.num_experts = config.get('num_experts', 64)
        self.top_k = config.get('num_experts_per_tok', 4)
        self.norm_topk_prob = config.get('norm_topk_prob', True)
        self.hidden_size = config.get('hidden_size', 4096)
        
        # 路由器权重
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_size))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算路由决策
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] 或 [num_tokens, hidden_size]
        
        Returns:
            router_logits: 原始路由 logits [num_tokens, num_experts]
            top_k_weights: Top-K 权重 [num_tokens, top_k]
            top_k_indices: Top-K 专家索引 [num_tokens, top_k]
        """
        # Reshape 为 [num_tokens, hidden_size]
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        
        # 计算路由 logits
        router_logits = F.linear(hidden_states_flat, self.weight)
        
        # Softmax 获取概率
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # 归一化 Top-K 权重
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        top_k_weights = top_k_weights.to(router_logits.dtype)
        
        return router_logits, top_k_weights, top_k_indices


class MoEExperts(nn.Module):
    """
    专家集合类
    使用 3D tensor 高效存储所有专家权重
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.num_experts = config.get('num_experts', 64)
        self.hidden_size = config.get('hidden_size', 4096)
        self.intermediate_size = config.get('intermediate_size', self.hidden_size * 4)
        self.act_fn = nn.SiLU()
        
        # 使用 3D tensor 存储所有专家的 gate_up_proj 和 down_proj 权重
        # gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        # down_proj: [num_experts, hidden_size, intermediate_size]
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 tokens 路由到专家并执行计算
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            top_k_indices: [num_tokens, top_k]
            top_k_weights: [num_tokens, top_k]
        
        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_indices.shape[1]
        
        # 初始化输出
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # 使用 one-hot 获取专家掩码
        # expert_mask: [num_experts, num_tokens, top_k]
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 1, 0)
            # 找出被选中的专家
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        # 对每个被选中的专家执行计算
        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0]
            if expert_idx >= self.num_experts:
                continue
            
            # 找出哪些 token 使用了这个专家
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            # 获取这些 token 的 hidden states
            current_state = hidden_states[token_idx]
            
            # 计算: gate_up_proj -> chunk(2) -> SiLU(gate) * up -> down_proj
            gate_up_output = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up_output.chunk(2, dim=-1)
            
            # SwiGLU 激活
            current_hidden_states = self.act_fn(gate) * up
            
            # down_proj
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            
            # 应用路由权重
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            
            # 累加到最终输出
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        
        return final_hidden_states


class StochasticMoELayer(nn.Module):
    """
    随机 MoE 层（带 GPU/CPU 混合执行）
    支持:
    - 路由计算（模拟开销）
    - 随机专家选择
    - GPU/CPU 混合执行
    - 预取机制
    """
    
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 配置参数
        self.hidden_size = config.get('hidden_size', 4096)
        self.total_experts = config.get('num_experts', 64)
        self.active_experts = config.get('num_experts_per_tok', 4)
        self.gpu_capacity = config.get('gpu_capacity', 8)
        self.prefetch_num = config.get('prefetch_num', 4)
        self.prefetch_acc = config.get('prefetch_acc', 0.8)
        
        # 1. 路由器（模拟路由计算开销）
        self.gate = TopKRouter(config)
        
        # 2. 专家集合
        self.experts = MoEExperts(config)
        
        # 3. 物理资源模拟
        # 模拟带宽消耗的 dummy buffer
        self.expert_param_size = self.hidden_size * (self.hidden_size * 4) * 2
        self.cpu_expert_data = torch.randn(
            self.expert_param_size // 2, dtype=torch.float16, device='cpu'
        )
        
        # 公用专家（GPU 端，用于测算力）
        self.common_gpu_expert = LlamaLikeExpert(self.hidden_size).cuda().half()
        
        # 不保存 CPU 专家作为模块（避免被 .cuda() 移动）
        
        # 预取流
        self.prefetch_stream = torch.cuda.Stream()
        
        # 4. 状态管理（缓存表）
        self.gpu_resident_set = set(
            random.sample(range(self.total_experts), min(self.gpu_capacity, self.total_experts))
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        use_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] 或 [num_tokens, hidden_size]
            use_router_logits: 是否使用路由器 logits（用于模拟计算开销）
        
        Returns:
            output: MoE 输出
        """
        # 保存原始形状
        orig_shape = hidden_states.shape
        num_tokens = hidden_states.view(-1, self.hidden_size).shape[0]
        
        # 1. 路由计算（模拟开销）
        if use_router_logits:
            # 实际计算路由 logits（模拟开销）
            router_logits, top_k_weights, top_k_indices = self.gate(hidden_states)
        else:
            # 随机选择专家
            num_tokens = hidden_states.view(-1, self.hidden_size).shape[0]
            top_k_indices = torch.randint(
                0, self.total_experts, (num_tokens, self.active_experts),
                device=hidden_states.device
            )
            top_k_weights = torch.ones_like(top_k_indices, dtype=hidden_states.dtype) / self.active_experts
            router_logits = None
        
        # 2. 根据命中率判断 GPU/CPU 执行
        hits = int(self.active_experts * self.prefetch_acc)
        misses = self.active_experts - hits
        
        # Reshape hidden_states 为 [num_tokens, hidden_size]
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        
        # 3. 执行专家计算
        if hits > 0:
            # GPU 命中部分：使用 GPU 专家
            gpu_hidden_states = self._execute_gpu_hits(hidden_states_flat, hits)
        else:
            gpu_hidden_states = torch.zeros_like(hidden_states_flat)
        
        if misses > 0:
            # GPU 未命中部分：使用 CPU 专家（带数据搬运惩罚）
            cpu_hidden_states = self._execute_cpu_miss(hidden_states_flat, misses)
        else:
            cpu_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # 合并结果
        output = gpu_hidden_states + cpu_hidden_states
        
        # 4. 触发预取
        if self.prefetch_num > 0:
            self._trigger_prefetch()
        
        # 恢复原始形状
        return output.view(orig_shape)
    
    def _execute_gpu_hits(
        self,
        x: torch.Tensor,
        hits: int,
    ) -> torch.Tensor:
        """
        执行 GPU 命中的专家
        
        Args:
            x: [num_tokens, hidden_size]
            hits: 命中次数
        
        Returns:
            output: GPU 计算结果
        """
        # 模拟计算：运行 hits 次公用专家
        output = x
        input_dtype = x.dtype
        for _ in range(hits):
            # 确保与 common_gpu_expert 的 dtype 一致（half）
            if output.dtype != torch.float16:
                output = output.to(torch.float16)
            output = self.common_gpu_expert(output)
            # 恢复原始 dtype
            if output.dtype != input_dtype:
                output = output.to(input_dtype)
        return output
    
    def _execute_cpu_miss(
        self,
        x: torch.Tensor,
        misses: int,
    ) -> torch.Tensor:
        """
        执行 CPU 未命中的专家（带数据搬运惩罚）
        
        Args:
            x: [num_tokens, hidden_size]
            misses: 未命中次数
        
        Returns:
            output: CPU 计算结果
        """
        # 惩罚 1: 数据从 GPU -> CPU（转换为 float32）
        x_cpu = x.cpu().float()
        
        # 惩罚 2: CPU 慢速计算（临时创建 CPU 专家）
        cpu_expert = LlamaLikeExpert(self.hidden_size).float().cpu()
        output_cpu = x_cpu
        for _ in range(misses):
            output_cpu = cpu_expert(output_cpu)
        
        # 惩罚 3: 结果从 CPU -> GPU（恢复原始 dtype）
        output_gpu = output_cpu.cuda().to(x.dtype)
        
        return output_gpu
    
    def _trigger_prefetch(self):
        """
        触发随机预取
        在后台流中模拟专家数据的预取
        """
        # 随机选择要预取的专家
        prefetch_candidates = random.sample(range(self.total_experts), self.prefetch_num)
        
        with torch.cuda.stream(self.prefetch_stream):
            for candidate in prefetch_candidates:
                if candidate not in self.gpu_resident_set:
                    # 模拟带宽消耗：将 expert 大小的 dummy 数据搬到 GPU
                    temp = self.cpu_expert_data.to('cuda', non_blocking=True)
                    
                    # 更新缓存表（随机替换策略）
                    if len(self.gpu_resident_set) >= self.gpu_capacity:
                        evicted = random.choice(list(self.gpu_resident_set))
                        self.gpu_resident_set.remove(evicted)
                    self.gpu_resident_set.add(candidate)
    
    def get_gpu_cache_size(self) -> int:
        """获取当前 GPU 缓存的专家数量"""
        return len(self.gpu_resident_set)

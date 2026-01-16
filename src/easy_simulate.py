import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from NsightProfiler import NsightProfiler

class LlamaLikeExpert(nn.Module):
    """
    标准的 SwiGLU 专家结构 (used in Llama, Mixtral, DeepSeek, etc.)
    包含 3 个线性层：Gate, Up, Down
    """
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        # 默认 intermediate_size 通常是 hidden_size 的 4 倍或者是 8/3 倍
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
            
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class StochasticMoELayer(nn.Module):
    def __init__(self, hidden_size, total_experts, active_experts, gpu_capacity, prefetch_num, prefetch_acc):
        """
        gpu_capacity: 显存能存多少个专家
        prefetch_num: 每次随机预取多少个专家
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.total_experts = total_experts
        self.active_experts = active_experts
        self.gpu_capacity = gpu_capacity
        self.prefetch_num = prefetch_num
        self.prefetch_acc = prefetch_acc
        
        # 1. 物理资源模拟，只是在模拟"传输"和"计算"的开销
        # 定义一个标准的专家大小用于模拟带宽消耗 (假设 FFN 是 4倍 hidden)
        self.expert_param_size = hidden_size * (hidden_size * 4) * 2 # 粗略估计参数量
        # 这是一个用于模拟搬运的 dummy buffer (CPU端)
        self.cpu_expert_data = torch.randn(self.expert_param_size // 2, dtype=torch.float16, device='cpu')
        
        # 模拟计算用的权重 (GPU端，公用一个，只为了测算力)
        self.common_gpu_expert = LlamaLikeExpert(hidden_size).cuda().half()
        
        # 模拟cpu计算的权重 (CPU端)
        self.common_cpu_expert = LlamaLikeExpert(hidden_size).half()
        
        self.attn_proj = nn.Linear(hidden_size, 3*hidden_size).cuda().half()    # 转移到GPU上且精度转为FP16
        self.o_proj = nn.Linear(hidden_size, hidden_size).cuda().half()
        # 2. 状态管理 (Set 模拟缓存表)
        # 初始随机填充显存
        self.gpu_resident_set = set(random.sample(range(total_experts), min(gpu_capacity, total_experts)))
        self.prefetch_stream = torch.cuda.Stream()

    # --- 1. Attention (固定开销) ---
    def _compute_attention(self, x):
        # x: [Batch, Seq, Hidden]
        batch, seq, dim = x.shape

        qkv = self.attn_proj(x)
        q,k,v = qkv.chunk(3, dim=-1)
        x_attn = F.scaled_dot_product_attention(
            q.view(batch, seq, 32, -1).transpose(1, 2),
            k.view(batch, seq, 32, -1).transpose(1, 2),
            v.view(batch, seq, 32, -1).transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

        x = x + self.o_proj(x_attn.transpose(1, 2).reshape(batch, seq, dim))

    def _execute_gpu_hits(self, x, hits):
        for _ in range(hits):
                _ = self.common_gpu_expert(x)
        return x
    
    def _execute_cpu_miss(self, x, misses):
        # 惩罚 1: 数据从 GPU -> CPU
        x_cpu = x.cpu() 
        # 惩罚 2: CPU 慢速计算
        for _ in range(misses):
            _ = self.common_cpu_expert(x_cpu)
        # 惩罚 3: 结果从 CPU -> GPU
        x_cpu = x_cpu.cuda()
        return x + x_cpu
    def _trigger_prefetch(self):
        # 模拟：随机决定下一时刻要搬谁进来
        prefetch_candidates = random.sample(range(self.total_experts), self.prefetch_num)
        
        with torch.cuda.stream(self.prefetch_stream):
            for candidate in prefetch_candidates:
                if candidate not in self.gpu_resident_set:
                    # 模拟带宽消耗：将 expert 大小的 dummy 数据搬到 GPU
                    # 注意：这会占用 PCIe 带宽，可能影响上面的 misses 搬运速度
                    temp = self.cpu_expert_data.to('cuda', non_blocking=True)
                    
                    # 更新缓存表 (简单的随机替换策略)
                    if len(self.gpu_resident_set) >= self.gpu_capacity:
                        # 随机踢出一个
                        evicted = random.choice(list(self.gpu_resident_set))
                        self.gpu_resident_set.remove(evicted)
                    self.gpu_resident_set.add(candidate)
        
    def forward(self, x):
        self._compute_attention(x)
        
        # --- 3. 命中检测与执行 ---
        hits = int(self.active_experts * self.prefetch_acc)
        misses = self.active_experts - hits
        
        # [优化] token路由到专家 ---> 专家加载计算
        # [路径 A]: GPU 命中 (直接计算)
        if hits > 0:
            # 模拟计算耗时：运行 len(hits) 次公用专家
            # 实际中是将 token 分组，这里简化为把 Batch 扩大模拟总 FLOPs
            self._execute_gpu_hits(x, hits)
        
        # [路径 B]: GPU 未命中 (CPU计算，但是数据需要临时搬运)
        if misses > 0: 
            self._execute_cpu_miss(x, misses)
            
        # --- 4. 随机预取 (Stochastic Prefetch) ---
        if self.prefetch_num > 0:
            self._trigger_prefetch()
        return x

# --- 评测脚本 ---
def run_simulation():
    # 配置参数
    config = {
        "hidden_size": 4096,
        "total_experts": 64,      # 总共有64个专家
        "active_experts": 4,      # 每一层用4个
        "gpu_capacity": 8,        # 显存很小，只能存8个 (高 Miss 率场景)
        "prefetch_num": 4,         # 预取下一层所需要的全部专家
        "prefetch_acc": 0.8
    }
    
    print(f"--- Setting: Capacity {config['gpu_capacity']}/{config['total_experts']} ---")
    model = StochasticMoELayer(**config)
    x = torch.randn(1, 256, 4096).cuda().half() # Batch=1, Seq=4096

    # 预热
    for _ in range(2): model(x)
    torch.cuda.synchronize()

    print("预热结束")
    # 运行 10 层 (模拟一个 10 层的模型)
    NsightProfiler.register_layer_hooks(model)
    
    # 手动劫持逻辑方法 (显示为自定义 Tag)
    NsightProfiler.wrap_method(model, '_compute_attention', tag_name='1_Attention_Phase')
    NsightProfiler.wrap_method(model, '_execute_gpu_hits',  tag_name='2_GPU_Expert_Hits')
    NsightProfiler.wrap_method(model, '_execute_cpu_miss',tag_name='⚠️_3_CPU_Fallback_Misses')
    NsightProfiler.wrap_method(model, '_trigger_prefetch',  tag_name='🌊_4_Async_Prefetch')
    
    start = time.time()
    for layer_idx in range(10):
        model(x)
        # 每一层结束后，同步流，确保预取完成（或者不同步以测试流水线效果）
        # 真实的 MoE Layer 之间是串行的，所以这里不用同步 prefetch stream，
        # 让它和下一层的 Attention 并行跑
        print(f"Layer {layer_idx + 1} 模拟结束")
    
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Total Latency (10 layers): {end - start:.4f} s")

if __name__ == "__main__":
    run_simulation()
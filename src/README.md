# Llama-like MoE LLM Simulator

一个模拟 Llama-like 架构（包含 MoE）的大型语言模型性能测试框架。

## 📁 项目结构

```
src/
├── cache_utils.py       # KV Cache 实现
├── attention.py         # Attention 模块（支持 RoPE、GQA、KV Cache）
├── moe.py              # MoE 模块（Router、Experts、混合 GPU/CPU 执行）
├── layer.py            # Decoder Layer（整合 Attention + MoE）
├── model.py            # 主模型和评测脚本
├── easy_simulate.py    # 入口脚本（兼容旧版）
└── README.md           # 本文件
```

## 🎯 功能特性

### 1. **KV Cache 支持**
- 完整的 KV Cache 管理
- 支持 Prefill 和 Decode 两种模式
- 按层存储和检索历史 K/V

### 2. **Attention 模块**
- RoPE（旋转位置编码）
- GQA（Grouped Query Attention）支持
- KV Cache 集成
- Flash Attention 优化（可选）

### 3. **MoE 模块**
- Top-K 路由器（模拟路由计算开销）
- 高效的专家执行（3D tensor 优化）
- **GPU/CPU 混合执行**
  - GPU 命中：直接计算
  - CPU 未命中：数据搬运 + CPU 计算 + 结果搬运
- 随机预取机制

### 4. **Decoder Layer**
- 标准 PreNorm 结构
- 双残差连接
- Attention → MoE 数据流串联

## 🚀 快速开始

### 运行完整测试

```bash
cd /home/shiyaochang/workspace/tasks/Llama-like-LLM-Simulatior/src
python easy_simulate.py
```

或直接运行主模型：

```bash
python model.py
```

### 输出示例

```
============================================================
Llama-like MoE LLM Simulator
============================================================
Hidden Size: 4096
Num Layers: 10
Num Experts: 64
Active Experts per Token: 4
GPU Capacity: 8/64
Prefetch Accuracy: 0.8
============================================================

============================================================
Prefill Simulation (Sequence Length: 256)
============================================================
预热中...
预热结束，开始测试...

Layer 1/10 完成 | GPU Cache: 8 experts
Layer 2/10 完成 | GPU Cache: 8 experts
...
Layer 10/10 完成 | GPU Cache: 8 experts

============================================================
Prefill 总耗时: 2.3456 s
平均每层耗时: 0.2346 s
============================================================

============================================================
Decode Simulation (Prefill: 256, Decode Steps: 10)
============================================================
Prefill 阶段：填充 256 tokens...
Prefill 完成，KV Cache 长度: 256

Decode 阶段：生成 10 tokens...
Step 1/10 | 耗时: 0.0123 s | KV Cache: 257 tokens
Step 2/10 | 耗时: 0.0115 s | KV Cache: 258 tokens
...
Step 10/10 | 耗时: 0.0118 s | KV Cache: 266 tokens

============================================================
Decode 总耗时: 0.1200 s
平均每 token 耗时: 0.0120 s
Tokens/s: 83.33
============================================================
```

## ⚙️ 配置参数

### 模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 4096 | 隐藏层维度 |
| `num_hidden_layers` | 10 | 模型层数 |
| `num_attention_heads` | 32 | 注意力头数 |
| `num_key_value_heads` | 32 | KV 头数（用于 GQA） |
| `max_position_embeddings` | 4096 | 最大序列长度 |
| `head_dim` | 128 | 每个 head 的维度 |
| `rope_theta` | 10000.0 | RoPE 基数 |

### MoE 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_experts` | 64 | 总专家数量 |
| `num_experts_per_tok` | 4 | 每个 token 激活的专家数 |
| `intermediate_size` | 16384 | MLP 中间维度 |
| `use_router_logits` | False | 是否使用路由器 logits（模拟开销） |

### 模拟配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gpu_capacity` | 8 | GPU 能缓存的专家数量 |
| `prefetch_num` | 4 | 每次预取的专家数量 |
| `prefetch_acc` | 0.8 | 预取命中率（决定 GPU/CPU 执行比例） |

## 📝 使用示例

### 1. 创建模型

```python
from model import LlamaLikeMoEModel

config = {
    "hidden_size": 4096,
    "num_hidden_layers": 10,
    "num_experts": 64,
    "num_experts_per_tok": 4,
    "gpu_capacity": 8,
    "prefetch_acc": 0.8,
    "use_moe": True,
}

model = LlamaLikeMoEModel(config)
model = model.cuda().half()
model.eval()
```

### 2. Prefill 模式

```python
from model import run_prefill_simulation

prefill_time = run_prefill_simulation(
    model=model,
    config=config,
    num_layers=10,
    seq_len=256,
    use_profiler=True,
)
```

### 3. Decode 模式

```python
from model import run_decode_simulation

decode_time = run_decode_simulation(
    model=model,
    config=config,
    num_layers=10,
    prefill_seq_len=256,
    decode_steps=10,
    use_profiler=True,
)
```

## 🔧 模块说明

### `cache_utils.py`

- `SimpleKVCache`: 简化的 KV Cache 实现
  - `update()`: 更新缓存
  - `get_seq_length()`: 获取序列长度
  - `get()`: 获取缓存的 K/V
  - `reset()`: 清空缓存

### `attention.py`

- `RotaryEmbedding`: RoPE 位置编码
- `LlamaLikeAttention`: Llama-like Attention 模块
  - 支持 KV Cache
  - 支持 GQA
  - Flash Attention 优化

### `moe.py`

- `LlamaLikeExpert`: 标准 SwiGLU 专家
- `TopKRouter`: Top-K 路由器
- `MoEExperts`: 专家集合（3D tensor 优化）
- `StochasticMoELayer`: 随机 MoE 层（支持 GPU/CPU 混合执行）

### `layer.py`

- `LlamaLikeRMSNorm`: RMS Normalization
- `LlamaLikeDecoderLayer`: 完整的 Decoder Layer
  - Attention + MoE
  - 双残差连接

### `model.py`

- `LlamaLikeMoEModel`: 主模型
- `run_prefill_simulation()`: Prefill 评测
- `run_decode_simulation()`: Decode 评测
- `main()`: 完整的测试流程

## 📊 性能分析

### 使用 Nsight Profiler

代码已集成 `NsightProfiler` 支持：

```python
# model.py 或 easy_simulate.py 中
use_profiler=True
```

### 监控指标

- Prefill 耗时
- Decode 耗时
- Tokens/s
- GPU Cache 命中情况
- KV Cache 大小

## 🎨 设计理念

### 参考模型

- **Qwen3 MoE**: 路由器设计、专家执行、KV Cache
- **DeepSeek V3**: MoE 架构、GQA、RoPE

### 模拟策略

- **路由计算**: 使用线性层模拟路由开销
- **专家选择**: 随机选择（不受路由器结果影响）
- **GPU/CPU 执行**: 根据命中率决定执行位置
- **数据搬运**: 使用 dummy buffer 模拟 PCIe 带宽消耗

## 📚 TODO

- [ ] 添加更多评测场景（不同序列长度、不同命中率）
- [ ] 支持多 batch 模拟
- [ ] 添加可视化工具
- [ ] 优化 CPU/GPU 混合执行策略
- [ ] 支持更复杂的 MoE 路由策略

## 📄 License

MIT License

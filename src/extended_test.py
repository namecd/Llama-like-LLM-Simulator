"""
扩展评测脚本 - 测试多种场景
包括：
1. 不同序列长度的 Prefill 测试
2. 不同命中率的 Decode 测试
3. 参数扫描
"""

import torch
import time
from model import LlamaLikeMoEModel, run_prefill_simulation, run_decode_simulation


def run_multi_scenario_test():
    """
    运行多场景测试
    """
    print("="*60)
    print("Llama-like MoE LLM Simulator - 多场景评测")
    print("="*60)

    # 基础配置
    base_config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 10,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "head_dim": 128,
        "rope_theta": 10000.0,
        
        # MoE 配置
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "intermediate_size": 16384,
        "norm_topk_prob": True,
        "use_router_logits": False,
        
        # 模拟配置
        "gpu_capacity": 8,
        "prefetch_num": 4,
        "use_moe": True,
    }

    # 创建模型
    model = LlamaLikeMoEModel(base_config)
    model = model.cuda().half()
    model.eval()

    # 场景 1: 不同序列长度的 Prefill 测试
    print("\n" + "="*60)
    print("场景 1: 不同序列长度的 Prefill 测试")
    print("="*60)

    seq_lengths = [64, 128, 256, 512, 1024]
    base_config["prefetch_acc"] = 0.8

    prefill_results = []

    for seq_len in seq_lengths:
        print(f"\n--- 序列长度: {seq_len} ---")
        try:
            prefill_time = run_prefill_simulation(
                model=model,
                config=base_config,
                num_layers=10,
                seq_len=seq_len,
                use_profiler=False,
            )
            prefill_results.append((seq_len, prefill_time))
            print(f"耗时: {prefill_time:.4f} s | Tokens/s: {seq_len / prefill_time:.2f}")
        except Exception as e:
            print(f"测试失败: {e}")

    # 场景 2: 不同命中率的 Decode 测试
    print("\n" + "="*60)
    print("场景 2: 不同命中率的 Decode 测试")
    print("="*60)

    hit_rates = [0.5, 0.7, 0.8, 0.9, 0.95]
    decode_results = []

    for hit_rate in hit_rates:
        config = base_config.copy()
        config["prefetch_acc"] = hit_rate

        print(f"\n--- 命中率: {hit_rate:.2f} ---")
        try:
            decode_time = run_decode_simulation(
                model=model,
                config=config,
                num_layers=10,
                prefill_seq_len=256,
                decode_steps=10,
                use_profiler=False,
            )
            tokens_per_sec = 10 / decode_time
            decode_results.append((hit_rate, decode_time, tokens_per_sec))
            print(f"耗时: {decode_time:.4f} s | Tokens/s: {tokens_per_sec:.2f}")
        except Exception as e:
            print(f"测试失败: {e}")

    # 场景 3: GPU 容量影响测试
    print("\n" + "="*60)
    print("场景 3: GPU 容量影响测试")
    print("="*60)

    gpu_capacities = [4, 8, 16, 32]
    capacity_results = []

    for gpu_cap in gpu_capacities:
        config = base_config.copy()
        config["gpu_capacity"] = gpu_cap
        config["prefetch_acc"] = 0.8

        print(f"\n--- GPU 容量: {gpu_cap}/{config['num_experts']} ---")
        try:
            decode_time = run_decode_simulation(
                model=model,
                config=config,
                num_layers=10,
                prefill_seq_len=256,
                decode_steps=10,
                use_profiler=False,
            )
            tokens_per_sec = 10 / decode_time
            capacity_results.append((gpu_cap, decode_time, tokens_per_sec))
            print(f"耗时: {decode_time:.4f} s | Tokens/s: {tokens_per_sec:.2f}")
        except Exception as e:
            print(f"测试失败: {e}")

    # 场景 4: 专家数量影响测试
    print("\n" + "="*60)
    print("场景 4: 每个 Token 激活的专家数影响测试")
    print("="*60)

    experts_per_toks = [2, 4, 6, 8]
    expert_results = []

    for num_experts_per_tok in experts_per_toks:
        config = base_config.copy()
        config["num_experts_per_tok"] = num_experts_per_tok
        config["prefetch_acc"] = 0.8

        print(f"\n--- 每个 Token 激活专家数: {num_experts_per_tok} ---")
        try:
            decode_time = run_decode_simulation(
                model=model,
                config=config,
                num_layers=10,
                prefill_seq_len=256,
                decode_steps=10,
                use_profiler=False,
            )
            tokens_per_sec = 10 / decode_time
            expert_results.append((num_experts_per_tok, decode_time, tokens_per_sec))
            print(f"耗时: {decode_time:.4f} s | Tokens/s: {tokens_per_sec:.2f}")
        except Exception as e:
            print(f"测试失败: {e}")

    # 总结报告
    print("\n" + "="*60)
    print("测试总结报告")
    print("="*60)

    print("\n1. 不同序列长度 Prefill 测试:")
    print("序列长度\t耗时(s)\tTokens/s")
    print("-"*40)
    for seq_len, t in prefill_results:
        print(f"{seq_len}\t\t{t:.4f}\t{seq_len/t:.2f}")

    print("\n2. 不同命中率 Decode 测试:")
    print("命中率\t\t耗时(s)\tTokens/s")
    print("-"*40)
    for hit_rate, t, tps in decode_results:
        print(f"{hit_rate:.2f}\t\t{t:.4f}\t{tps:.2f}")

    print("\n3. GPU 容量影响测试:")
    print("GPU容量\t耗时(s)\tTokens/s")
    print("-"*40)
    for gpu_cap, t, tps in capacity_results:
        print(f"{gpu_cap}\t\t{t:.4f}\t{tps:.2f}")

    print("\n4. 专家数量影响测试:")
    print("激活专家数\t耗时(s)\tTokens/s")
    print("-"*40)
    for num_experts, t, tps in expert_results:
        print(f"{num_experts}\t\t{t:.4f}\t{tps:.2f}")

    print("="*60)


def run_batch_simulation_test():
    """
    测试多 batch 模拟
    """
    print("\n" + "="*60)
    print("多 Batch 模拟测试")
    print("="*60)

    config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 10,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "head_dim": 128,
        "rope_theta": 10000.0,
        
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "intermediate_size": 16384,
        "norm_topk_prob": True,
        "use_router_logits": False,
        
        "gpu_capacity": 8,
        "prefetch_num": 4,
        "prefetch_acc": 0.8,
        "use_moe": True,
    }

    model = LlamaLikeMoEModel(config)
    model = model.cuda().half()
    model.eval()

    batch_sizes = [1, 2, 4, 8]

    print("\n--- Prefill 多 Batch 测试 (seq_len=256) ---")
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        # 由于当前实现不支持多 batch，这里使用单 batch 模拟
        # 实际多 batch 支持需要修改模型架构
        total_time = 0
        for i in range(batch_size):
            try:
                prefill_time = run_prefill_simulation(
                    model=model,
                    config=config,
                    num_layers=10,
                    seq_len=256,
                    use_profiler=False,
                )
                total_time += prefill_time
            except Exception as e:
                print(f"  批次 {i+1} 失败: {e}")

        print(f"  总耗时: {total_time:.4f} s")
        print(f"  平均耗时: {total_time/batch_size:.4f} s/batch")
        print(f"  Tokens/s: {(256 * batch_size) / total_time:.2f}")

    print("\n--- Decode 多 Batch 测试 (prefill=256, decode=10) ---")
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        total_time = 0
        for i in range(batch_size):
            try:
                decode_time = run_decode_simulation(
                    model=model,
                    config=config,
                    num_layers=10,
                    prefill_seq_len=256,
                    decode_steps=10,
                    use_profiler=False,
                )
                total_time += decode_time
            except Exception as e:
                print(f"  批次 {i+1} 失败: {e}")

        print(f"  总耗时: {total_time:.4f} s")
        print(f"  平均耗时: {total_time/batch_size:.4f} s/batch")
        print(f"  Tokens/s: {(10 * batch_size) / total_time:.2f}")

    print("="*60)


if __name__ == "__main__":
    # 运行多场景测试
    run_multi_scenario_test()

    # 运行多 batch 模拟测试
    run_batch_simulation_test()

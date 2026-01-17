"""
Llama-like MoE LLM Simulator (Legacy Script)
已迁移到模块化设计，请使用 model.py

保留此文件用于向后兼容
"""

# 从新模块导入
from model import LlamaLikeMoEModel, run_prefill_simulation, run_decode_simulation

# --- 评测脚本（使用新模型）---
def run_simulation():
    """
    使用新的模块化设计运行模拟
    """
    # 配置参数
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
        
        # MoE 配置
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "intermediate_size": 16384,
        "norm_topk_prob": True,
        "use_router_logits": False,
        
        # 模拟配置
        "gpu_capacity": 8,
        "prefetch_num": 4,
        "prefetch_acc": 0.8,
        "use_moe": True,
    }
    
    print(f"--- Setting: Capacity {config['gpu_capacity']}/{config['num_experts']} ---")
    print(f"--- Prefetch Accuracy: {config['prefetch_acc']} ---")
    
    # 创建模型
    model = LlamaLikeMoEModel(config)
    model = model.cuda().half()
    model.eval()
    
    # 运行 Prefill 模拟
    print("\n" + "="*60)
    print("Running Prefill Simulation")
    print("="*60)
    prefill_time = run_prefill_simulation(
        model=model,
        config=config,
        num_layers=10,
        seq_len=256,
        use_profiler=True,
    )
    
    # 运行 Decode 模拟
    print("\n" + "="*60)
    print("Running Decode Simulation")
    print("="*60)
    decode_time = run_decode_simulation(
        model=model,
        config=config,
        num_layers=10,
        prefill_seq_len=256,
        decode_steps=10,
        use_profiler=True,
    )
    
    # 总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Prefill (256 tokens): {prefill_time:.4f} s")
    print(f"Decode (10 tokens): {decode_time:.4f} s")
    print(f"Decode Tokens/s: {10 / decode_time:.2f}")
    print("="*60)


if __name__ == "__main__":
    run_simulation()
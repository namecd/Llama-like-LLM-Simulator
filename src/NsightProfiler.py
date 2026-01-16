import torch
import torch.cuda.nvtx as nvtx
import functools
import json
import time
class NsightProfiler:
    """
    非侵入式 Nsight 性能分析注入工具
    """
    
    @staticmethod
    def wrap_method(instance, method_name, tag_name=None):
        """
        劫持实例的方法，在执行前后加入 NVTX 和计时
        """
        if not hasattr(instance, method_name):
            print(f"Warning: Method {method_name} not found in {instance}.")
            return

        original_method = getattr(instance, method_name)
        tag = tag_name or f"{instance.__class__.__name__}.{method_name}"

        @functools.wraps(original_method)
        def wrapper(*args, **kwargs):
            # 1. 标记开始
            # torch.cuda.synchronize() # 确保之前的 GPU 任务完成，时间更准
            start_time = time.perf_counter()
            nvtx.range_push(f"[{tag}]")
            
            try:
                # 2. 执行原方法
                result = original_method(*args, **kwargs)
                return result
            finally:
                # 3. 标记结束
                # torch.cuda.synchronize()
                end_time = time.perf_counter()
                nvtx.range_pop()
                # 可选：打印耗时，方便控制台查看，不影响 Nsight
                # print(f"⏱️  [Time] {tag}: {(end_time - start_time) * 1000:.2f} ms")

        # 动态替换实例的方法
        setattr(instance, method_name, wrapper)
        print(f"✅ Instrumented method: {method_name}")

    @staticmethod
    def register_layer_hooks(torch_model):
        """
        给 PyTorch 模型的每一层注册 Hook，能在 Nsight 里看到层级结构
        只监控 Attention 和 MoE 相关的关键计算层
        """
        def pre_hook(module, input, name):
            nvtx.range_push(f"L:{name}")

        def post_hook(module, input, output, name):
            nvtx.range_pop()

        # 定义关键词白名单
        # 包含这些词的层才会被监控，避免图表过于杂乱
        target_keywords = (
            "self_attn",        # Attention
            "mla",              # deepseek-V2 Attention
            "block_sparse_moe", # Mixtral MoE
            "mlp",               # deepseek-V2 MoE
            "attn",
            "attention",
            "moe",
            "expert",
            "gate"
        )

        count = 0
        # 遍历所有子模块（层）
        for name, module in torch_model.named_modules():
            if not name: continue
            
            lname = name.lower()
            # 过滤逻辑
            if any(k in lname for k in target_keywords):
                # 使用 functools.partial 将 name 传进去
                module.register_forward_pre_hook(functools.partial(pre_hook, name=name))
                module.register_forward_hook(functools.partial(post_hook, name=name))
                count += 1
        
        print(f"✅ Registered Nsight hooks for {count} internal layers (Attention/MoE).")
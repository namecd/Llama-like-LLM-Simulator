"""
基础测试脚本
验证各个模块是否能够正常工作
"""
import torch
from cache_utils import SimpleKVCache
from attention import RotaryEmbedding, apply_rotary_pos_emb
from moe import LlamaLikeExpert, TopKRouter, MoEExperts
from layer import LlamaLikeRMSNorm, LlamaLikeDecoderLayer
from model import LlamaLikeMoEModel


def test_cache_utils():
    """测试 KV Cache"""
    print("Testing Cache Utils...")
    
    config = {
        'num_hidden_layers': 2,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'hidden_size': 512,
        'head_dim': 16,
    }
    
    cache = SimpleKVCache(config)
    
    # 测试 update
    batch_size = 2
    seq_len = 10
    num_kv_heads = 4
    head_dim = 16
    
    key_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim).cuda()
    value_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim).cuda()
    
    all_keys, all_values = cache.update(key_states, value_states, 0, {'cache_position': torch.arange(seq_len).cuda()})
    
    assert all_keys.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert all_values.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    assert cache.get_seq_length(0) == seq_len
    
    print("✓ Cache Utils test passed")


def test_attention():
    """测试 Attention 模块"""
    print("Testing Attention...")
    
    config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'max_position_embeddings': 1024,
        'rope_theta': 10000.0,
        'head_dim': 64,
    }
    
    # 测试 RoPE
    rotary_emb = RotaryEmbedding(config)
    x = torch.randn(2, 10, 512).cuda()
    position_ids = torch.arange(10).unsqueeze(0).expand(2, -1).cuda()
    
    cos, sin = rotary_emb(x, position_ids)
    assert cos.shape == (2, 10, 64)
    assert sin.shape == (2, 10, 64)
    
    print("✓ Attention test passed")


def test_moe():
    """测试 MoE 模块"""
    print("Testing MoE...")
    
    config = {
        'hidden_size': 512,
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'intermediate_size': 2048,
    }
    
    # 测试 Expert
    expert = LlamaLikeExpert(512, 2048).cuda()  # 保持 float32
    x = torch.randn(10, 512).cuda()
    output = expert(x)
    assert output.shape == (10, 512)
    
    # 测试 Router
    router = TopKRouter(config).cuda()
    router_logits, top_k_weights, top_k_indices = router(x)
    assert top_k_indices.shape == (10, 2)
    assert top_k_weights.shape == (10, 2)
    
    # 测试 MoEExperts
    experts = MoEExperts(config).cuda()
    hidden_states = torch.randn(10, 512).cuda()
    output = experts(hidden_states, top_k_indices, top_k_weights)
    assert output.shape == (10, 512)
    
    print("✓ MoE test passed")


def test_layer():
    """测试 Layer 模块"""
    print("Testing Layer...")
    
    config = {
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'max_position_embeddings': 1024,
        'rope_theta': 10000.0,
        'head_dim': 64,
        'rms_norm_eps': 1e-6,
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'intermediate_size': 2048,
        'gpu_capacity': 4,
        'prefetch_num': 2,
        'prefetch_acc': 0.8,
    }
    
    # 测试 RMS Norm
    norm = LlamaLikeRMSNorm(512)
    x = torch.randn(2, 10, 512)
    output = norm(x)
    assert output.shape == (2, 10, 512)
    
    # 测试 Decoder Layer
    layer = LlamaLikeDecoderLayer(config, layer_idx=0, use_moe=True).cuda().half()
    hidden_states = torch.randn(2, 10, 512).cuda().half()
    position_ids = torch.arange(10).unsqueeze(0).expand(2, -1).cuda()
    
    output = layer(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        cache_position=torch.arange(10).cuda(),  # long 类型
        use_router_logits=False,
    )
    
    assert output.shape == (2, 10, 512)
    
    print("✓ Layer test passed")


def test_model():
    """测试主模型"""
    print("Testing Model...")
    
    config = {
        'vocab_size': 1000,
        'hidden_size': 512,
        'num_hidden_layers': 2,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'max_position_embeddings': 1024,
        'rms_norm_eps': 1e-6,
        'head_dim': 64,
        'rope_theta': 10000.0,
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'intermediate_size': 2048,
        'norm_topk_prob': True,
        'use_router_logits': False,
        'gpu_capacity': 4,
        'prefetch_num': 2,
        'prefetch_acc': 0.8,
        'use_moe': True,  # 打开 MoE 测试
    }
    
    model = LlamaLikeMoEModel(config)
    model = model.cuda().half()  # half 精度
    model.eval()
    
    # 测试 forward (prefill)
    input_ids = torch.randint(0, 1000, (2, 10)).cuda()
    hidden_states, past_key_values = model(
        input_ids=input_ids,
        use_cache=True,
        use_router_logits=False,
    )
    
    assert hidden_states.shape == (2, 10, 512)
    assert past_key_values is not None
    assert past_key_values.get_seq_length() == 10
    
    # 测试 decode
    input_ids_decode = torch.randint(0, 1000, (2, 1)).cuda()
    hidden_states_decode, past_key_values_decode = model(
        input_ids=input_ids_decode,
        past_key_values=past_key_values,
        use_cache=True,
        use_router_logits=False,
    )
    
    assert hidden_states_decode.shape == (2, 1, 512)
    assert past_key_values_decode.get_seq_length() == 11
    
    print("✓ Model test passed")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Running Basic Tests")
    print("="*60 + "\n")
    
    try:
        test_cache_utils()
        test_attention()
        test_moe()
        test_layer()
        test_model()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

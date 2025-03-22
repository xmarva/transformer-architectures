import pytest
import torch
from src.modules.attention import MultiHeadAttention


def test_attention_output_shape():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    q = k = v = torch.randn(batch_size, seq_len, d_model)
    
    output = mha(q, k, v)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_attention_with_mask():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    q = k = v = torch.randn(batch_size, seq_len, d_model)
    
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)
    
    output_with_mask = mha(q, k, v, mask)
    
    assert output_with_mask.shape == (batch_size, seq_len, d_model)


def test_multi_head_division():
    d_model = 512
    num_heads = 7  # Not divisible by 512
    
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model, num_heads)


def test_scaled_dot_product_attention():
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64
    
    mha = MultiHeadAttention(d_model=num_heads*head_dim, num_heads=num_heads)
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    output = mha.scaled_dot_product_attention(Q, K, V)
    
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)
import pytest
import torch
from src.modules.feed_forward import FeedForward


def test_feed_forward_output_shape():
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    ffn = FeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ffn(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_feed_forward_activations():
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    ffn = FeedForward(d_model, d_ff)
    x = torch.ones(batch_size, seq_len, d_model)
    
    intermediate = ffn.linear1(x)
    assert intermediate.shape == (batch_size, seq_len, d_ff)
    
    intermediate_activated = torch.relu(intermediate)
    assert torch.all(intermediate_activated >= 0)


def test_feed_forward_dropout():
    batch_size = 2
    seq_len = 10
    d_model = 512
    dropout = 0.5 
    
    ffn = FeedForward(d_model, dropout=dropout)
    x = torch.ones(batch_size, seq_len, d_model)
    
    ffn.train()
    
    outputs = [ffn(x) for _ in range(5)]
    
    assert not all(torch.allclose(outputs[0], output) for output in outputs[1:])
    
    ffn.eval()
    eval_outputs = [ffn(x) for _ in range(5)]
    assert all(torch.allclose(eval_outputs[0], output) for output in eval_outputs[1:])
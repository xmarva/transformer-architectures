import pytest
import torch
from src.modules.encoder import EncoderLayer, Encoder


def test_encoder_layer_output_shape():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    encoder_layer = EncoderLayer(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = encoder_layer(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_encoder_layer_with_mask():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    encoder_layer = EncoderLayer(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, -2:] = 0 
    
    output = encoder_layer(x, mask)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_encoder_multiple_layers():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    num_layers = 3
    
    encoder = Encoder(num_layers, d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert len(encoder.layers) == num_layers


def test_encoder_residual_connection():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    encoder_layer = EncoderLayer(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    x_copy = x.clone()
    
    output = encoder_layer(x)
    
    # Output should be different from input due to transformations
    assert not torch.allclose(output, x)
    
    # But due to residual connections, it shouldn't be completely different
    # This is a simple heuristic: output should be closer to input than to random tensor
    random_tensor = torch.randn_like(x)
    input_diff = (output - x_copy).abs().mean().item()
    random_diff = (output - random_tensor).abs().mean().item()
    
    assert input_diff < random_diff
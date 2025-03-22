import pytest
import torch
from src.modules.positional_encoding import PositionalEncoding


def test_positional_encoding_output_shape():
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    pe = PositionalEncoding(d_model, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_positional_encoding_adds_position_info():
    batch_size = 1
    seq_len = 10
    d_model = 512
    
    pe = PositionalEncoding(d_model, dropout=0.0)  # No dropout for deterministic test
    x = torch.zeros(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    # Different positions should have different encodings
    assert not torch.allclose(output[:, 0, :], output[:, 1, :])
    
    # Same positions in different batches should have the same encoding
    if batch_size > 1:
        assert torch.allclose(output[0, 0, :], output[1, 0, :])


def test_positional_encoding_preserves_input():
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    pe = PositionalEncoding(d_model, dropout=0.0)  # No dropout for deterministic test
    x = torch.ones(batch_size, seq_len, d_model)
    
    output = pe(x)
    
    # The positional encoding adds values but doesn't scale the original input
    # So we should check that the difference between output and input
    # matches the positional encoding that was added
    pe_added = output - x
    expected_pe = pe.pe[:, :seq_len, :]
    
    expected_pe = expected_pe.expand(batch_size, -1, -1)
    
    # Use a small tolerance for floating point differences
    assert torch.allclose(pe_added, expected_pe, rtol=1e-5, atol=1e-5)
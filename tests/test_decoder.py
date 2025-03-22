import pytest
import torch
from src.modules.decoder import DecoderLayer, Decoder


def test_decoder_layer_output_shape():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    decoder_layer = DecoderLayer(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    
    output = decoder_layer(x, enc_output)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_decoder_layer_with_masks():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    decoder_layer = DecoderLayer(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len+5, d_model)  
    
    src_mask = torch.ones(batch_size, 1, 1, seq_len+5)
    src_mask[:, :, :, -3:] = 0 
    
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
    tgt_mask = tgt_mask.expand(batch_size, 1, seq_len, seq_len)
    
    output = decoder_layer(x, enc_output, src_mask, tgt_mask)
    
    assert output.shape == (batch_size, seq_len, d_model)


def test_decoder_multiple_layers():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    num_layers = 3
    
    decoder = Decoder(num_layers, d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    
    output = decoder(x, enc_output)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert len(decoder.layers) == num_layers


def test_decoder_cross_attention():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    decoder_layer = DecoderLayer(d_model, num_heads)
    
    x = torch.zeros(batch_size, seq_len, d_model)
    enc_output = torch.ones(batch_size, seq_len, d_model)
    
    output = decoder_layer(x, enc_output)
    
    assert not torch.allclose(output, torch.zeros_like(output))
    
    diff_enc_output = enc_output * 2
    diff_output = decoder_layer(x, diff_enc_output)
    
    assert not torch.allclose(output, diff_output)
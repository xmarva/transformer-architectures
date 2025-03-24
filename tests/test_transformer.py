import unittest
import torch
import math
from typing import Tuple

from src.modules.transformer import Transformer
from src.utils.masks import create_padding_mask, create_causal_mask, create_masks
"""
class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1200
        self.src_seq_len = 10
        self.tgt_seq_len = 12
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1
        self.pad_idx = 0
        
        self.src = torch.randint(1, self.src_vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt = torch.randint(1, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_len))
        
        self.src[:, -2:] = self.pad_idx
        self.tgt[:, -2:] = self.pad_idx
        
    def test_transformer_initialization(self):
        transformer = Transformer(
            self.src_vocab_size, 
            self.tgt_vocab_size,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.d_ff,
            self.dropout
        )
        
        self.assertEqual(transformer.d_model, self.d_model)
        self.assertEqual(transformer.encoder_embedding.num_embeddings, self.src_vocab_size)
        self.assertEqual(transformer.decoder_embedding.num_embeddings, self.tgt_vocab_size)
        self.assertEqual(len(transformer.encoder.layers), self.num_layers)
        self.assertEqual(len(transformer.decoder.layers), self.num_layers)
        
    def test_transformer_forward_pass(self):
        src_mask = create_padding_mask(self.src, self.pad_idx)
        
        tgt_padding_mask = create_padding_mask(self.tgt, self.pad_idx)
        tgt_causal_mask = create_causal_mask(self.tgt.size(1)).to(self.tgt.device)
        tgt_causal_mask = tgt_causal_mask.bool()
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        transformer = Transformer(
            self.src_vocab_size, 
            self.tgt_vocab_size,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.d_ff,
            self.dropout
        )
        
        output = transformer(self.src, self.tgt, src_mask, tgt_mask)
        
        expected_shape = (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_shared_embeddings(self):
        shared_vocab_size = 1000
        
        transformer = Transformer(
            shared_vocab_size, 
            shared_vocab_size,
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.d_ff,
            self.dropout,
            share_embeddings=True
        )
        
        self.assertIs(transformer.encoder_embedding, transformer.decoder_embedding)
        
        src = torch.randint(1, shared_vocab_size, (self.batch_size, self.src_seq_len))
        tgt = torch.randint(1, shared_vocab_size, (self.batch_size, self.tgt_seq_len))
        
        src[:, -2:] = self.pad_idx
        tgt[:, -2:] = self.pad_idx
        
        src_mask = create_padding_mask(src, self.pad_idx)
        
        tgt_padding_mask = create_padding_mask(tgt, self.pad_idx)
        tgt_causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
        tgt_causal_mask = tgt_causal_mask.bool() 
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        output = transformer(src, tgt, src_mask, tgt_mask)
        expected_shape = (self.batch_size, self.tgt_seq_len, shared_vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_create_padding_mask(self):
        seq = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 2, 3, 4, 0],
            [1, 2, 3, 4, 5]
        ])
        
        mask = create_padding_mask(seq, pad_idx=0)
        
        expected_shape = (4, 1, 1, 5)
        self.assertEqual(mask.shape, expected_shape)
        
        expected_mask = torch.tensor([
            [[[[1, 1, 1, 0, 0]]]],
            [[[[1, 1, 0, 0, 0]]]],
            [[[[1, 1, 1, 1, 0]]]],
            [[[[1, 1, 1, 1, 1]]]]
        ], dtype=torch.bool)
        
        self.assertTrue(torch.all(mask == expected_mask))
        
    def test_create_causal_mask(self):
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        expected_shape = (1, 1, seq_len, seq_len)
        self.assertEqual(mask.shape, expected_shape)
        
        expected_mask = torch.tensor([
            [[[
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1]
            ]]]
        ], dtype=torch.float)
        
        self.assertTrue(torch.all(mask == expected_mask))
        
    def test_create_masks_modified(self):
        src = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 2, 0, 0, 0]
        ])
        
        tgt = torch.tensor([
            [1, 2, 0, 0],
            [1, 2, 3, 0]
        ])
        
        src_mask = create_padding_mask(src, pad_idx=0)
        
        tgt_padding_mask = create_padding_mask(tgt, pad_idx=0)
        tgt_causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
        tgt_causal_mask = tgt_causal_mask.bool()
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        self.assertEqual(src_mask.shape, (2, 1, 1, 5))
        self.assertEqual(tgt_mask.shape, (2, 1, 4, 4))
        
        expected_src_mask = torch.tensor([
            [[[[1, 1, 1, 0, 0]]]],
            [[[[1, 1, 0, 0, 0]]]]
        ], dtype=torch.bool)
        
        self.assertTrue(torch.all(src_mask == expected_src_mask))
        
        expected_tgt_causal_component = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ], dtype=torch.bool)
        
        for i in range(4):
            for j in range(4):
                if not expected_tgt_causal_component[i, j]:
                    self.assertFalse(tgt_mask[0, 0, i, j])
        
    def test_transformer_training_mode(self):
        transformer = Transformer(
            self.src_vocab_size, 
            self.tgt_vocab_size,
            self.d_model,
            self.num_heads
        )
        
        self.assertTrue(transformer.training)
        
        transformer.eval()
        self.assertFalse(transformer.training)
        
        transformer.train()
        self.assertTrue(transformer.training)
        
    def test_parameter_initialization(self):
        transformer = Transformer(
            self.src_vocab_size, 
            self.tgt_vocab_size,
            self.d_model,
            self.num_heads,
            num_layers=2 
        )
        
        for name, param in transformer.named_parameters():
            if param.dim() > 1: 
                self.assertGreater(param.std().item(), 0)
                
                self.assertAlmostEqual(param.mean().item(), 0, delta=0.1)
                
    def test_fixed_positional_embeddings(self):
        transformer = Transformer(
            self.src_vocab_size,
            self.tgt_vocab_size
        )
        
        self.assertIn('positional_encoding.pe', dict(transformer.named_buffers()))
        self.assertNotIn('positional_encoding.pe', dict(transformer.named_parameters()))
        
"""
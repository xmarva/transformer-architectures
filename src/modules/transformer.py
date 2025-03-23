import torch
import torch.nn as nn
from typing import Optional
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        d_model: int = 512, 
        num_heads: int = 8, 
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        share_embeddings: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        
        if share_embeddings:
            self.decoder_embedding = self.encoder_embedding
        else:
            self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
            
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None, 
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        src_emb = self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model))
        enc_output = self.encoder(src_emb, src_mask)
        
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model))
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        return output
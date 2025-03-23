import torch
from typing import Tuple


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Creates a mask for padding tokens.
    
    Args:
        seq: Input sequence tensor of shape [batch_size, seq_len]
        pad_idx: Index used for padding
        
    Returns:
        Mask tensor of shape [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Creates a causal mask for decoder self-attention.
    
    Args:
        seq_len: Length of the sequence
        
    Returns:
        Causal mask of shape [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
    return mask


def create_masks(
    src: torch.Tensor, 
    tgt: torch.Tensor, 
    pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Creates both source and target masks.
    
    Args:
        src: Source sequence of shape [batch_size, src_len]
        tgt: Target sequence of shape [batch_size, tgt_len]
        pad_idx: Index used for padding
        
    Returns:
        Tuple of (src_mask, tgt_mask)
    """
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    tgt_causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
    
    # Combine padding and causal masks
    tgt_mask = tgt_padding_mask & tgt_causal_mask
    
    return src_mask, tgt_mask
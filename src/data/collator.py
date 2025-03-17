import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from .tokenization.base import BaseTokenizer

@dataclass
class TranslationCollator:
    src_tokenizer: BaseTokenizer
    tgt_tokenizer: BaseTokenizer
    max_length: int = 64
    pad_token: str = "<PAD>"

    def __call__(self, batch: List[Dict]) -> Dict[str, np.ndarray]:
        src_texts = [item['src'] for item in batch]
        tgt_texts = [item['tgt'] for item in batch]
        
        src_tokens = [self.src_tokenizer.tokenize(text) for text in src_texts]
        tgt_tokens = [self.tgt_tokenizer.tokenize(text) for text in tgt_texts]
        
        return {
            'src_tokens': self._pad_sequences(src_tokens, self.src_tokenizer),
            'tgt_tokens': self._pad_sequences(tgt_tokens, self.tgt_tokenizer),
            'src_mask': self._create_mask(src_tokens),
            'tgt_mask': self._create_mask(tgt_tokens)
        }

    def _pad_sequences(self, sequences: List[List[int]], tokenizer: BaseTokenizer) -> np.ndarray:
        pad_id = tokenizer.token2id[self.pad_token]
        padded = np.full((len(sequences), self.max_length), pad_id, dtype=np.int64)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_length)
            padded[i, :length] = seq[:length]
        return padded

    def _create_mask(self, sequences: List[List[int]]) -> np.ndarray:
        mask = np.zeros((len(sequences), self.max_length), dtype=np.int64)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_length)
            mask[i, :length] = 1
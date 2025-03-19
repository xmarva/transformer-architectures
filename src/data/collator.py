import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Collator:
    src_tokenizer: Any
    tgt_tokenizer: Any
    max_length: int = 128
    pad_token_id: int = 0
    
    def __call__(self, batch: List[Dict]) -> Dict[str, np.ndarray]:
        src_batch = []
        tgt_batch = []
        
        for item in batch:
            src = item['src_ids'][:self.max_length]
            tgt = item['tgt_ids'][:self.max_length]
            
            src_batch.append(self._pad_sequence(src, self.src_tokenizer))
            tgt_batch.append(self._pad_sequence(tgt, self.tgt_tokenizer))
        
        return {
            'src_ids': np.array(src_batch),
            'tgt_ids': np.array(tgt_batch),
            'src_mask': (np.array(src_batch) != self.pad_token_id).astype(np.float32),
            'tgt_mask': (np.array(tgt_batch) != self.pad_token_id).astype(np.float32)
        }
    
    def _pad_sequence(self, sequence: List[int], tokenizer: Any) -> List[int]:
        pad_length = self.max_length - len(sequence)
        padding = [tokenizer.token2id.get("<PAD>", self.pad_token_id)] * pad_length
        return sequence + padding if pad_length > 0 else sequence
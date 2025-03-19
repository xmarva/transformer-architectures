from transformers import AutoTokenizer
from .base import BaseTokenizer
from typing import List, Dict

class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            language=kwargs.get('language', 'multi'),
            vocab_size=kwargs.get('vocab_size', 50000),
            special_tokens=kwargs.get('special_tokens', [])
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **kwargs
        )
        self._update_vocab_mappings()
        
    def _update_vocab_mappings(self):
        self.token2id = self.tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        
    def train(self, texts: List[str]) -> None:
        pass 
        
    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
        
    def batch_tokenize(self, texts: List[str]) -> Dict:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_tensors=None
        )
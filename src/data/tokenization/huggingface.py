from transformers import AutoTokenizer
from .base import BaseTokenizer

class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        super().__init__(special_tokens=[])
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._build_base_vocab()
    
    def _build_base_vocab(self) -> None:
        self.token2id = self.hf_tokenizer.get_vocab()
        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def tokenize(self, text: str) -> List[int]:
        return self.hf_tokenizer.encode(
            text, 
            add_special_tokens=True, 
            truncation=True,
            max_length=self.hf_tokenizer.model_max_length
        )
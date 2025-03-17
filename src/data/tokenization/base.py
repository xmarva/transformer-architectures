from dataclasses import dataclass
from typing import List, Dict
from collections import Counter
import re

@dataclass
class BaseTokenizer:
    language: str
    vocab_size: int
    min_freq: int = 2
    special_tokens: List[str] = None
    
    def __post_init__(self) -> None:
        self.special_tokens = self.special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}
    
    def preprocess_text(self, text: str) -> List[str]:
        return re.findall(r"\w+[\w']*|['’][a-z]+|[^\w\s]", text.lower())
    
    def get_stats(self, examples: List[str]) -> Counter:
        counter = Counter()
        for text in examples:
            tokens = self.preprocess_text(text)
            counter.update(tokens)
        return counter
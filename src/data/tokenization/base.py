from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Counter
import re

@dataclass
class BaseTokenizer(ABC):
    language: str
    vocab_size: int
    special_tokens: List[str] = None
    
    def __post_init__(self):
        self.special_tokens = self.special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2token: Dict[int, str] = {idx: token for idx, token in enumerate(self.special_tokens)}
    
    def preprocess_text(self, text: str) -> List[str]:
        tokens = re.findall(r"\w+[\w']*|['’][a-z]+|[^\w\s]", text.lower())
        return tokens
    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass
    
    def get_stats(self, examples: List[str]) -> Counter[str]:
        counter = Counter()
        for text in examples:
            tokens = self.preprocess_text(text)
            counter.update(tokens)
        return counter
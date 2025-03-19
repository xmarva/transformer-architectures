from abc import ABC, abstractmethod
from typing import Dict, List
from collections import Counter
import re

class BaseTokenizer(ABC):
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}
    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass
    
    def preprocess_text(self, text: str) -> List[str]:
        return re.findall(r"\w+[\w']*|['’][a-z]+|[^\w\s]", text.lower())
    
    def get_stats(self, examples: List[str]) -> Counter:
        return Counter(chain(*[self.preprocess_text(t) for t in examples]))
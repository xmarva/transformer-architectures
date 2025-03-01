from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self._build_vocab()
        
    def _build_vocab(self):
        self.token2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.special_tokens)}
        
    @abstractmethod
    def train(self, texts):
        pass
    
    @abstractmethod
    def tokenize(self, text):
        pass
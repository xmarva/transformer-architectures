import pytest
from src.data.tokenization.base import BaseTokenizer

class ConcreteTokenizer(BaseTokenizer):
    def train(self, texts):
        self.vocab = list(set(" ".join(texts)))
        
    def tokenize(self, text):
        return [self.token2id.get(c, self.token2id["<UNK>"]) for c in text]

@pytest.fixture
def tokenizer():
    return ConcreteTokenizer(special_tokens=["<PAD>", "<UNK>"])

def test_base_tokenizer_initialization(tokenizer):
    assert len(tokenizer.token2id) == 2
    assert tokenizer.token2id["<PAD>"] == 0
    assert tokenizer.id2token[1] == "<UNK>"

def test_vocab_building():
    tok = ConcreteTokenizer(special_tokens=["a", "b", "c"])
    assert tok.token2id == {"a": 0, "b": 1, "c": 2}
    assert tok.id2token == {0: "a", 1: "b", 2: "c"}
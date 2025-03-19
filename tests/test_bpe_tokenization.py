import pytest
from src.data.tokenization.bpe import BPETokenizer

class TestBPETokenizer:
    @pytest.fixture
    def tokenizer(self):
        return BPETokenizer(language='en', vocab_size=1000)
    
    def test_train(self, tokenizer):
        texts = ["hello world", "world test"]
        tokenizer.train(texts, num_merges=3)
        assert len(tokenizer.merges) == 3
        assert len(tokenizer.token2id) > 4
    
    def test_tokenize(self, tokenizer):
        texts = ["hello world"]
        tokenizer.train(texts, num_merges=2)
        encoded = tokenizer.tokenize("hello world")
        assert encoded[0] == tokenizer.token2id['<BOS>']
        assert encoded[-1] == tokenizer.token2id['<EOS>']
        assert len(encoded) > 2
import pytest
from unittest.mock import patch, MagicMock
from src.data.tokenization.base import BaseTokenizer

class TestBaseTokenizer:
    @patch.multiple(BaseTokenizer, __abstractmethods__=set())
    def test_initialization(self):
        tokenizer = BaseTokenizer(language='en', vocab_size=1000)
        assert tokenizer.token2id["<PAD>"] == 0
        assert tokenizer.id2token[1] == "<UNK>"
    
    @patch.multiple(BaseTokenizer, __abstractmethods__=set())
    def test_preprocess_text(self):
        tokenizer = BaseTokenizer(language='en', vocab_size=1000)
        text = "Hello! How's it going?"
        tokens = tokenizer.preprocess_text(text)
        assert tokens == ["hello", "!", "how's", "it", "going", "?"]
    
    @patch.multiple(BaseTokenizer, __abstractmethods__=set())
    def test_get_stats(self):
        tokenizer = BaseTokenizer(language='en', vocab_size=1000)
        stats = tokenizer.get_stats(["hello world", "world test"])
        assert stats["hello"] == 1
        assert stats["world"] == 2
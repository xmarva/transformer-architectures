import pytest
from data.tokenization import BaseTokenizer, BPETokenizer
from typing import List

@pytest.fixture
def base_tokenizer() -> BaseTokenizer:
    return BaseTokenizer(language="en", vocab_size=1000)

def test_base_tokenizer_initialization(base_tokenizer):
    assert base_tokenizer.token2id["<PAD>"] == 0
    assert base_tokenizer.id2token[3] == "<EOS>"
    
def test_preprocessing(base_tokenizer):
    text = "Hello! This is a Test."
    tokens = base_tokenizer.preprocess_text(text)
    assert tokens == ["hello", "!", "this", "is", "a", "test", "."]
    
def test_token_stats(base_tokenizer):
    stats = base_tokenizer.get_stats(["test test", "another example"])
    assert stats["test"] == 2

@pytest.fixture
def trained_bpe() -> BPETokenizer:
    tokenizer = BPETokenizer(language="en", vocab_size=1000)
    texts = ["apple banana orange", "banana orange apple"]
    tokenizer.train(texts, num_merges=10)
    return tokenizer

def test_bpe_training(trained_bpe):
    assert len(trained_bpe.merges) == 10
    assert len(trained_bpe.token2id) > 15

def test_bpe_tokenization(trained_bpe):
    tokens = trained_bpe.tokenize("applepie")
    assert tokens[0] == trained_bpe.token2id["<BOS>"]
    assert tokens[-1] == trained_bpe.token2id["<EOS>"]
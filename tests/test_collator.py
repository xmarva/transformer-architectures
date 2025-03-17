import pytest
import numpy as np
from data.collator import TranslationCollator
from data.tokenization import BPETokenizer

@pytest.fixture
def test_collator():
    src_tokenizer = BPETokenizer(language="en", vocab_size=1000)
    tgt_tokenizer = BPETokenizer(language="ru", vocab_size=1000)
    texts = ["test text", "another example"]
    src_tokenizer.train(texts, num_merges=5)
    tgt_tokenizer.train(texts, num_merges=5)
    return TranslationCollator(src_tokenizer, tgt_tokenizer, max_length=10)

def test_collator_shapes(test_collator):
    batch = [
        {"src": "Hello world", "tgt": "Привет мир"},
        {"src": "Test sentence", "tgt": "Тестовое предложение"}
    ]
    processed = test_collator(batch)
    
    assert processed["src_tokens"].shape == (2, 10)
    assert processed["tgt_mask"].shape == (2, 10)
    
def test_padding_logic(test_collator):
    test_sequence = [1, 2, 3]
    padded = test_collator._pad_sequences([test_sequence], test_collator.src_tokenizer)
    assert padded[0, 3] == test_collator.src_tokenizer.token2id["<PAD>"]
    
def test_mask_creation(test_collator):
    sequences = [[1,2,3], [1,2]]
    mask = test_collator._create_mask(sequences)
    assert mask.tolist() == [[1,1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0]]
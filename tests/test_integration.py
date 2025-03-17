import pytest
from data import DatasetFactory, BPETokenizer, TranslationCollator

@pytest.fixture
def full_pipeline():
    dataset = DatasetFactory.create("tatoeba")
    src_tokenizer = BPETokenizer(language="en", vocab_size=1000)
    tgt_tokenizer = BPETokenizer(language="ru", vocab_size=1000)
    
    # Train on small subset
    train_texts = [item["src"] for item in dataset[:100]]
    src_tokenizer.train(train_texts, num_merges=50)
    tgt_tokenizer.train(train_texts, num_merges=50)
    
    return TranslationCollator(src_tokenizer, tgt_tokenizer)

def test_end_to_end(full_pipeline):
    test_batch = [
        {"src": "Test input", "tgt": "Тестовая строка"},
        {"src": "Another example", "tgt": "Другой пример"}
    ]
    
    processed = full_pipeline(test_batch)
    
    assert processed["src_tokens"].dtype == np.int64
    assert processed["tgt_mask"].sum() > 4
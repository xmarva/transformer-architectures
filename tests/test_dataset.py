import pytest
from src.data.dataset import TranslationDataset

@pytest.fixture
def sample_dataset():
    return TranslationDataset(
        name="Helsinki-NLP/tatoeba",
        lang1="en",
        lang2="ru"
    )
    
def test_dataset_loading(sample_dataset):
    assert sample_dataset.data is not None
    assert len(sample_dataset) > 0
    
def test_dataset_structure(sample_dataset):
    sample = sample_dataset[0]
    assert isinstance(sample, dict)
    assert "en" in sample
    assert "ru" in sample
    assert isinstance(sample["en"], str)
    assert isinstance(sample["ru"], str)
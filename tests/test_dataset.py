import pytest
from data.dataset import TatoebaDataset, DatasetFactory
from datasets import DatasetDict

@pytest.fixture
def sample_dataset() -> TatoebaDataset:
    return DatasetFactory.create("tatoeba", lang_pair=("en", "ru"))

def test_dataset_loading(sample_dataset):
    assert len(sample_dataset) > 0
    sample = sample_dataset[0]
    assert "src" in sample
    assert "tgt" in sample
    assert isinstance(sample["src"], str)
    assert isinstance(sample["tgt"], str)

def test_dataset_factory():
    dataset = DatasetFactory.create("tatoeba", lang_pair=("fr", "de"))
    assert dataset.lang_pair == ("fr", "de")
    
def test_invalid_dataset():
    with pytest.raises(ValueError):
        DatasetFactory.create("invalid_dataset")
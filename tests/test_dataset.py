# File: tests/test_dataset_factory.py
import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset
from src.data.dataset_factory import DatasetFactory

class TestDatasetFactory:
    @patch('src.data.dataset_factory.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        mock_dataset = MagicMock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset
        
        dataset = DatasetFactory.load_dataset("test_dataset", version="1.0")
        
        mock_load_dataset.assert_called_once_with("test_dataset", version="1.0")
        assert dataset == mock_dataset
    
    @patch('src.data.dataset_factory.load_dataset')
    def test_load_dataset_error(self, mock_load_dataset):
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        with pytest.raises(ValueError) as excinfo:
            DatasetFactory.load_dataset("nonexistent_dataset")
        
        assert "Failed to load dataset nonexistent_dataset" in str(excinfo.value)
    
    @patch.object(DatasetFactory, 'load_dataset')
    def test_load_translation_dataset(self, mock_load_dataset):
        mock_dataset = MagicMock(spec=Dataset)
        mock_load_dataset.return_value = mock_dataset
        
        result = DatasetFactory.load_translation_dataset()
        
        mock_load_dataset.assert_called_once_with("Helsinki-NLP/tatoeba", lang1="en", lang2="ru")
        assert result == mock_dataset
    
    def test_preprocess_dataset(self):
        mock_dataset = MagicMock(spec=Dataset)
        mock_tokenizer = MagicMock()
        
        mock_tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]
        
        mock_dataset.map.return_value = MagicMock(spec=Dataset)
        
        result = DatasetFactory.preprocess_dataset(mock_dataset, mock_tokenizer, "text", 3)
        
        mock_dataset.map.assert_called_once()
        
        map_function = mock_dataset.map.call_args[0][0]
        example = {"text": "sample text"}
        processed = map_function(example)
        
        assert processed["input_ids"] == [1, 2, 3]
        assert processed["attention_mask"] == [1, 1, 1]
        mock_tokenizer.tokenize.assert_called_with("sample text")
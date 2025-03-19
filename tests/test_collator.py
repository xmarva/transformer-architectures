import pytest
import numpy as np
from unittest.mock import MagicMock
from src.data.collator import Collator

class TestCollator:
    @pytest.fixture
    def mock_tokenizers(self):
        src_tokenizer = MagicMock()
        src_tokenizer.token2id = {"<PAD>": 0}
        
        tgt_tokenizer = MagicMock()
        tgt_tokenizer.token2id = {"<PAD>": 0}
        
        return src_tokenizer, tgt_tokenizer
    
    def test_collator_padding(self, mock_tokenizers):
        src_tokenizer, tgt_tokenizer = mock_tokenizers
        collator = Collator(src_tokenizer, tgt_tokenizer, max_length=10)
        
        batch = [
            {'src_ids': [1, 2, 3], 'tgt_ids': [4, 5]},
            {'src_ids': [6, 7, 8, 9], 'tgt_ids': [10, 11, 12]}
        ]
        
        processed = collator(batch)
        
        # Check shapes
        assert processed['src_ids'].shape == (2, 10)
        assert processed['tgt_ids'].shape == (2, 10)
        assert processed['src_mask'].shape == (2, 10)
        assert processed['tgt_mask'].shape == (2, 10)
        
        # Check padding
        assert processed['src_ids'][0].tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        assert processed['src_ids'][1].tolist() == [6, 7, 8, 9, 0, 0, 0, 0, 0, 0]
        assert processed['tgt_ids'][0].tolist() == [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        assert processed['tgt_ids'][1].tolist() == [10, 11, 12, 0, 0, 0, 0, 0, 0, 0]
        
        # Check masks
        assert processed['src_mask'][0].sum() == 3
        assert processed['src_mask'][1].sum() == 4
        assert processed['tgt_mask'][0].sum() == 2
        assert processed['tgt_mask'][1].sum() == 3
        
        # Verify mask values directly
        assert processed['src_mask'].sum(axis=1).tolist() == [3, 4] 
        assert processed['tgt_mask'].sum(axis=1).tolist() == [2, 3]
    
    def test_collator_truncation(self, mock_tokenizers):
        src_tokenizer, tgt_tokenizer = mock_tokenizers
        collator = Collator(src_tokenizer, tgt_tokenizer, max_length=5)
        
        batch = [
            {'src_ids': [1, 2, 3, 4, 5, 6, 7], 'tgt_ids': [8, 9, 10, 11, 12, 13]},
        ]
        
        processed = collator(batch)
        
        # Check truncation
        assert processed['src_ids'][0].tolist() == [1, 2, 3, 4, 5]
        assert processed['tgt_ids'][0].tolist() == [8, 9, 10, 11, 12]
        
        # Check masks after truncation
        assert processed['src_mask'][0].sum() == 5
        assert processed['tgt_mask'][0].sum() == 5
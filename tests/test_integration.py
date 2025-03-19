import pytest
import yaml
import os
from src.main import main
from src.data.dataset_factory import DatasetFactory

class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        config = {
            "tokenizer_type": "bpe",
            "tokenizer_params": {
                "src": {"language": "en", "vocab_size": 1000},
                "tgt": {"language": "ru", "vocab_size": 1000}
            },
            "collator_params": {"max_length": 50},
            "pipeline_params": {"batch_size": 16}
        }
        
        os.makedirs("src/configs", exist_ok=True)
        
        config_path = "src/configs/base_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        main() 
        dataset = DatasetFactory.load_translation_dataset()
        assert len(dataset['train']) > 0
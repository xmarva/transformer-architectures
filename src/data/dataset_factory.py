from typing import Dict, List
from datasets import Dataset, load_dataset

class DatasetFactory:
    @staticmethod
    def load_dataset(name: str, **kwargs) -> Dataset:
        try:
            kwargs["trust_remote_code"] = True
            dataset = load_dataset(name, **kwargs)
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset {name}: {str(e)}")
    
    @classmethod
    def load_translation_dataset(cls, name: str = "Helsinki-NLP/tatoeba", 
                                lang1: str = "en", lang2: str = "ru") -> Dict:
        dataset = cls.load_dataset(name, lang1=lang1, lang2=lang2, trust_remote_code=True)
        return dataset
    
    @staticmethod
    def preprocess_dataset(dataset: Dataset, tokenizer, text_field: str, max_length: int) -> Dataset:
        def process_example(example):
            tokens = tokenizer.tokenize(example[text_field])
            return {
                'input_ids': tokens[:max_length],
                'attention_mask': [1]*len(tokens[:max_length])
            }
        return dataset.map(process_example, batched=False)
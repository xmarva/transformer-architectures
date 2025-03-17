from datasets import load_dataset
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, str]:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

class TatoebaDataset(BaseDataset):
    def __init__(self, lang_pair: tuple = ('en', 'ru')):
        self.lang1, self.lang2 = lang_pair
        self.data = self._load_data()
    
    def _load_data(self) -> Any:
        return load_dataset(
            "Helsinki-NLP/tatoeba",
            lang1=self.lang1,
            lang2=self.lang2,
            trust_remote_code=True
        )['train']
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]['translation']
        return {'src': item[self.lang1], 'tgt': item[self.lang2]}

class DatasetFactory:
    @staticmethod
    def create(name: str, **kwargs) -> BaseDataset:
        if name == 'tatoeba':
            return TatoebaDataset(**kwargs)
        raise ValueError(f"Unknown dataset: {name}")
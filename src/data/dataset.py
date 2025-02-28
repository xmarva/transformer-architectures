from typing import Any
from datasets import load_dataset

class TranslationDataset:
    def __init__(self, name: str, lang1: str, lang2: str):
        """_summary_

        Args:
            name (str): Dataset identifier (e.g. 'Helsinki-NLP/tatoeba')
            lang1 (str): Source language code
            lang2 (str): Target language code
        """
        
        self.data = self._load_dataset(name, lang1, lang2)
        
    def _load_dataset(self, name: str, lang1: str, lang2: str) -> Any:
        
        print("Loading {name} dataset...")
        
        try:
            dataset = load_dataset(name, lang1=lang1, lang2=lang2, trust_remote_code=True)
            print("Dataset loaded successfully")
            
        except Exception as e:
            print(f"Error while loading dataset: {e}")
            
        return dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data["train"][idx]["translation"]
from datasets import Dataset
from typing import Callable, Dict

class TranslationDataPipeline:
    def __init__(
        self,
        tokenizer_fn: Callable,
        collator_fn: Callable,
        batch_size: int = 32
    ):
        self.tokenizer_fn = tokenizer_fn
        self.collator_fn = collator_fn
        self.batch_size = batch_size

    def __call__(self, dataset: Dataset) -> Dict:
        tokenized = dataset.map(
            self.tokenizer_fn,
            batched=True,
            batch_size=self.batch_size
        )
        return tokenized.with_format(
            'numpy',
            columns=['src_ids', 'tgt_ids']
        )
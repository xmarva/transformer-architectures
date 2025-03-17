from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm
from .base import BaseTokenizer

@dataclass
class BPETokenizer(BaseTokenizer):
    merges: Dict[Tuple[str, str], str] = None
    vocab: set = None

    def __post_init__(self):
        super().__post_init__()
        self.merges = self.merges or {}
        self.vocab = set(self.special_tokens)
    
    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        return [(word[i], word[i+1]) for i in range(len(word)-1)]
    
    def train(self, texts: List[str], num_merges: int) -> None:
        word_freqs = defaultdict(int)
        all_chars = set()
        
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                chars = list(token)
                word_freqs[' '.join(chars)] += 1
                all_chars.update(chars)
        
        for char in sorted(all_chars):
            if char not in self.token2id:
                idx = len(self.token2id)
                self.token2id[char] = idx
                self.id2token[idx] = char
        
        for _ in tqdm(range(num_merges), desc="BPE Training"):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self._merge_pair(best_pair, word_freqs)
    
    def _get_pairs(self, word_freqs) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs) -> None:
        merged = ''.join(pair)
        new_word_freqs = {}
        
        for word, freq in word_freqs.items():
            new_word = word.replace(' '.join(pair), merged)
            new_word_freqs[new_word] = freq
        
        self.merges[pair] = merged
        self.token2id[merged] = len(self.token2id)
        self.id2token[len(self.id2token)] = merged
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.token2id['<BOS>']]
        for token in self.preprocess_text(text):
            chars = list(token)
            while len(chars) > 1:
                pairs = self.get_pairs(chars)
                pair = next((p for p in pairs if p in self.merges), None)
                if not pair: break
                chars = self._apply_merge(chars, pair)
            tokens.extend([self.token2id.get(c, self.token2id['<UNK>']) for c in chars])
        tokens.append(self.token2id['<EOS>'])
        return tokens
    
    def _apply_merge(self, chars: List[str], pair: Tuple[str, str]) -> List[str]:
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars)-1 and (chars[i], chars[i+1]) == pair:
                new_chars.append(self.merges[pair])
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        return new_chars
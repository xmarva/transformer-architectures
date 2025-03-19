from typing import List, Dict, Tuple, Counter, DefaultDict, Set
from collections import defaultdict
from dataclasses import dataclass
from abc import abstractmethod
from src.data.tokenization.base import BaseTokenizer

@dataclass
class BPETokenizer(BaseTokenizer):
    merges: Dict[Tuple[str, str], str] = None
    vocab: Set[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.merges = {}
        self.vocab = set(self.special_tokens)
    
    def get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        return [(word[i], word[i+1]) for i in range(len(word)-1)]
    
    def train(self, texts: List[str], num_merges: int = 1000) -> None:
        word_freqs = defaultdict(int)
        all_chars = set()
        
        for text in texts:
            tokens = self.preprocess_text(text)
            for token in tokens:
                chars = list(token)
                word_freqs[' '.join(chars)] += 1
                all_chars.update(chars)
        
        for char in sorted(all_chars):
            self._add_token_to_vocab(char)
        
        for _ in range(num_merges):  # Fixed asterisk with underscore
            pair_freqs = self._count_pairs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self._merge_pair(best_pair, word_freqs)
    
    def _add_token_to_vocab(self, token: str) -> None:  # Fixed method name formatting
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            self.vocab.add(token)
    
    def _count_pairs(self, word_freqs: DefaultDict[str, int]) -> DefaultDict[Tuple[str, str], int]:  # Fixed method name formatting
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            pairs = self.get_pairs(symbols)
            for pair in pairs:
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: DefaultDict[str, int]) -> None:  # Fixed method name formatting
        new_token = ''.join(pair)
        self.merges[pair] = new_token
        self._add_token_to_vocab(new_token)
        
        new_word_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            new_word = word.replace(' '.join(pair), new_token)
            new_word_freqs[new_word] += freq
        
        word_freqs.clear()
        word_freqs.update(new_word_freqs)
    
    def tokenize(self, text: str) -> List[int]:
        tokens = self.preprocess_text(text)
        encoded = [self.token2id['<BOS>']]
        
        for token in tokens:
            current_symbols = list(token)
            while len(current_symbols) > 1:
                pairs = self.get_pairs(current_symbols)
                best_pair = None
                for pair in pairs:
                    if pair in self.merges:
                        best_pair = pair
                        break
                
                if not best_pair:
                    break
                
                merged = self.merges[best_pair]
                current_symbols = self._apply_merge(current_symbols, best_pair, merged)
            
            for symbol in current_symbols:
                encoded.append(self.token2id.get(symbol, self.token2id['<UNK>']))
        
        encoded.append(self.token2id['<EOS>'])
        return encoded
    
    def _apply_merge(self, symbols: List[str], pair: Tuple[str, str], merged: str) -> List[str]:  # Fixed method name formatting
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols)-1 and (symbols[i], symbols[i+1]) == pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols
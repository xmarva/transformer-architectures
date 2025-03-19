import pytest
from unittest.mock import patch, MagicMock
from src.data.tokenization.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

class TestHuggingFaceTokenizer:
    @pytest.fixture
    def mock_tokenizer(self):
        mock = MagicMock()
        mock.get_vocab.return_value = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "hello": 3,
            "world": 4
        }
        mock.encode.return_value = [0, 3, 4, 2]
        mock.return_value = {
            "input_ids": [[0, 3, 4, 2], [0, 4, 2]]
        }
        return mock

    @pytest.fixture
    def tokenizer(self, mock_tokenizer):
        with patch("src.data.tokenization.huggingface.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            tokenizer = HuggingFaceTokenizer(
                model_name="Helsinki-NLP/opus-mt-en-ru",
                src_lang="en",
                tgt_lang="ru"
            )
            return tokenizer

    def test_initialization(self, tokenizer, mock_tokenizer):
        assert tokenizer.language == 'multi'
        assert tokenizer.vocab_size == 5
        assert tokenizer.tokenizer == mock_tokenizer
        assert tokenizer.token2id["<s>"] == 0
        assert tokenizer.id2token[2] == "</s>"

    def test_tokenization(self, tokenizer, mock_tokenizer):
        text = "Hello world!"
        tokens = tokenizer.tokenize(text)
        mock_tokenizer.encode.assert_called_once_with(
            text,
            add_special_tokens=True,
            truncation=False
        )
        assert tokens == [0, 3, 4, 2]
        assert tokens[0] == tokenizer.token2id["<s>"]
        assert tokens[-1] == tokenizer.token2id["</s>"]

    def test_batch_tokenize(self, tokenizer, mock_tokenizer):
        texts = ["Hello world!", "World"]
        result = tokenizer.batch_tokenize(texts)
        mock_tokenizer.assert_called_once_with(
            texts,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        assert result == {"input_ids": [[0, 3, 4, 2], [0, 4, 2]]}

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 5

    def test_update_vocab_mappings(self, tokenizer, mock_tokenizer):
        mock_tokenizer.get_vocab.return_value = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "hello": 3,
            "world": 4,
            "new": 5
        }
        tokenizer._update_vocab_mappings()
        assert tokenizer.vocab_size == 6
        assert tokenizer.token2id["new"] == 5
        assert tokenizer.id2token[5] == "new"
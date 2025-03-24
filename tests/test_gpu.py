import pytest
import torch

@pytest.mark.gpu
def test_cuda_available():
    assert torch.cuda.is_available()
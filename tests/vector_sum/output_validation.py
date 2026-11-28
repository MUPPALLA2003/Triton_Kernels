import pytest
import torch
from src.triton_addition.naive_vector_sum import NaiveVectorSum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason="needs CUDA")
sizes = [2**20,2**24,2**26,2**24 + 3]

@pytest.mark.parametrize("n",sizes)
def test_vector_add_big(n):

    a = torch.randn(n,device="cuda")
    b = torch.randn(n,device="cuda")
    out = NaiveVectorSum.naive_vector_sum_wrapper(a,b)
    assert torch.allclose(out,a + b)
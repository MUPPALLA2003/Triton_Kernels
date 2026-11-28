import torch
import triton
import triton.language as tl

@triton.jit
def naive_vector_sum_kernel(a_ptr,b_ptr,out_ptr):

    pid = tl.program_id(axis=0)
    a_ptr_offset = a_ptr + pid
    b_ptr_offset = b_ptr + pid
    a = tl.load(a_ptr_offset)
    b = tl.load(b_ptr_offset)
    out_ptr_offset = out_ptr + pid
    out = a + b
    tl.store(out_ptr_offset,out)

class NaiveVectorSum:

    def naive_vector_sum_wrapper(a,b):

        output = torch.empty_like(a)
        n_elements = len(a)
        grid = (n_elements,)
        naive_vector_sum_kernel[grid](a,b,output)

        return output    
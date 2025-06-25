import torch
import triton
import triton.language as tl

@triton.jit
def t_log_softmax_fwd_k(x_ptr,
                    output_ptr,
                    input_row_stride,
                    output_row_stride,
                    n_rows,
                    n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    num_stages: tl.constexpr,
                    # NOTE: `constexpr` so it can be used as a shape value. <- TODO T: think about it
                    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages): # TODO T: it fails if I add stages??
        x_row_start_ptr = x_ptr + row_idx * input_row_stride
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_row_start_ptr + offsets, mask=mask, other=-1e9)
        x_minus_max = x - tl.max(x, axis=0)
        log_denominator = tl.exp(x_minus_max)
        log_denominator = tl.sum(log_denominator, axis=0)
        log_denominator = tl.log(log_denominator)
        output = x_minus_max - log_denominator
        # In case I want to change semantic to t_softmax_fwd_k:
        # nominator = tl.exp(x_minus_max)
        # denominator = tl.sum(nominator, axis=0)
        # output = nominator/denominator    
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + offsets, output, mask=mask)
    
def t_log_softmax_fwd_t(x: torch.Tensor):
    x_2d = x.reshape((-1, x.shape[-1])) # TODO T: without this reshape, this func is 2times faster
    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 
    output = torch.empty_like(x_2d)
    num_programs = min(n_rows, 100) # TODO T: compute correct number based on occupancy/SM
    t_log_softmax_fwd_k[(num_programs, 1, 1)](x_2d, output, x_2d.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE, 1)
    return output.reshape(x.shape)

aa = torch.randn((8, 12, 512, 512), device="cuda")
res = t_log_softmax_fwd_t(aa)
print(f'res', res.shape, res[0][0])
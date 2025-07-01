import sys
sys.path.append('../')
import torch
from model_torch_func import layernorm_fwd
#from model_triton import t_layernorm_bkwd2_p_t, t_layernorm_bkwd2_x_t

dloss_dx = torch.randn((8, 512, 768), device="cuda")
layer_params = (torch.ones((768), device="cuda"), torch.zeros((768), device="cuda")) # real init
#layer_params = (torch.randn((768), device="cuda"), torch.randn((768), device="cuda")) # unreal init
aa = torch.randn((8, 512, 768), device="cuda")


# dx
(_, vjpfunc) = torch.func.vjp(layernorm_fwd, layer_params, aa)
res1 = vjpfunc(dloss_dx)[1]
print(res1.shape, res1[-2:, -4:, -8:])

import triton
import triton.language as tl

# Note that the kernel assumes that n_cols < BLOCK_SIZE
# TODO T: investigate numerical differences from torch.func implementation
@triton.jit
def t_layernorm_bkwd2_x_k(dloss_dx_ptr,
                    param1_ptr,
                    x_ptr,
                    output_ptr,
                    dloss_dx_stride,
                    x_row_stride,
                    output_row_stride,
                    n_rows,
                    n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    num_stages: tl.constexpr,
                    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    # Load shared params
    # TODO T: I think triton will load them once into shared memory -> confirm
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    param1 = tl.load(param1_ptr + offsets, mask=mask, other=0.0)  
        
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages):
        dloss_dx_row_start_ptr = dloss_dx_ptr + row_idx * dloss_dx_stride
        dloss_dx = tl.load(dloss_dx_row_start_ptr + offsets, mask=mask, other=0.0)
        dloss_dx = dloss_dx * param1
        x_row_start_ptr = x_ptr + row_idx * x_row_stride    
        x = tl.load(x_row_start_ptr + offsets, mask=mask, other=0.0)
        
        # compute mean and std for x
        x_sum = tl.sum(x, axis=0)
        x_mu = x_sum/ n_cols
        x_minus_mu = x - x_mu
        x_minus_mu2 = x_minus_mu * x_minus_mu
        x_minus_mu2_sum = tl.sum(x_minus_mu2, axis=0)
        x_sigma2 = x_minus_mu2_sum / (n_cols-1)
        x_sigma = tl.sqrt_rn(x_sigma2)
        
        # normalize x
        x_norm = x_minus_mu/x_sigma    
        
        # bkwd quantities
        dloss_dx_sum = tl.sum(dloss_dx, axis=0)
        dloss_dx_mu = dloss_dx_sum/n_cols
        dloss_dx_x_norm = dloss_dx * x_norm
        dloss_dx_x_norm_sum = tl.sum(dloss_dx_x_norm, axis=0)
        dloss_dx_x_norm_mu = dloss_dx_x_norm_sum/n_cols
        
        n_adj = n_cols/(n_cols-1) # adjust for estimated vs calculated sigma
        output = dloss_dx - dloss_dx_mu - x_norm * dloss_dx_x_norm_mu * n_adj
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + offsets, output, mask=mask)
    
def t_layernorm_bkwd2_x_t(dloss_dx:torch.Tensor, layer_params: torch.Tensor, x: torch.Tensor):
    # TODO T: without this reshape, this func is 2times faster?
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    x_2d = x.reshape((-1, x.shape[-1])) 
    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 
    print(f'XXX', BLOCK_SIZE, n_cols)
    output = torch.empty_like(x_2d)
    # TODO T: The below numbers were tuned for A10 by choosing num_warps=8
    num_warps = 8
    num_stages = 2
    num_programs = min(n_rows, 480) 
    t_layernorm_bkwd2_x_k[(num_programs,)](dloss_dx_2d, layer_params[0], x_2d, output, 
                                       dloss_dx_2d.stride(0), x_2d.stride(0), output.stride(0), n_rows, n_cols, 
                                       BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
    return output.reshape(dloss_dx.shape)

res2 = t_layernorm_bkwd2_x_t(dloss_dx, layer_params, aa)
print(res2.shape, res2[-2:, -4:, -8:])

atol, rtol=1e-1, 1e-3
def pre_allclose(input, other, name, atol, rtol):
    allclose = torch.abs(input - other) - atol - rtol * torch.abs(other)
    print(name, torch.max(allclose))
pre_allclose(res1[0], res2[0], 'dx', atol=atol, rtol=rtol)
assert torch.allclose(res1, res2, atol=atol, rtol=rtol), (res1.shape, res2.shape, res1[:2, :4, :8], res2[:2, :4, :8])

import os
os.environ["TRITON_INTERPRET"] = "1"


import sys
sys.path.append('../')

import torch
#from model_torch_func import layernorm_fwd
#from model_triton import t_layernorm_bkwd2_p_t, t_layernorm_bkwd2_x_t

BS, N, D = 1, 1, 4 #8, 512, 768
dloss_dx = torch.tensor([[[ 0.7227,  0.6544, -0.7753,  0.5889]]], device="cuda")
#dloss_dx = torch.randn((BS, N, D), device="cuda")
layer_params = (torch.ones((D), device="cuda"), torch.zeros((D), device="cuda"))
aa = torch.tensor([[[-0.2614,  0.9828, -0.1427,  0.1716]]], device="cuda")
#aa = torch.randn((BS, N, D), device="cuda")

print(f'dloss_dx', dloss_dx)
print(f'aa', aa)


print(f'\n### TORCH_FUNC')
# dx
def layernorm_fwd(layer_params, x):
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    print("x_mean", x_mean)
    x_std = torch.std(x, axis=-1, keepdims=True) # TODO XXX: Compute variance, add epsilon and take a sqrt instead (in order to avoid division by zero)
    print("x_std", x_std)
    normalized_x = (x - x_mean) / x_std
    print("normalized_x", normalized_x)
    return torch.multiply(normalized_x, layer_params[0][None, :]) + layer_params[1][None, :] # since both layer_params are output_dim x

(_, vjpfunc) = torch.func.vjp(layernorm_fwd, layer_params, aa)
res1 = vjpfunc(dloss_dx)[1]
print("RESULT:", res1.shape, res1[-2:, -4:, -8:])

print(f'\n### t_layernorm_bkwd2_x')
#from model_triton import t_layernorm_bkwd2_x
# TODO XXX XXX: investigate why this is more memory efficient than my implementation above
# (Inspired from llm.c)
def normalized_x_bkwd2_plus(dloss_dx, x): # d [(x-x_mean)/x_std] / dx
    # f(x) = x - x_mean, g(x) = x_std
    BS, N = x.shape
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_rstd = 1/torch.std(x, axis=-1, keepdims = True)
    x_norm = (x - x_mean) * x_rstd
    
    n_adj = N/(N-1)
    dloss_dx = dloss_dx - dloss_dx.mean(-1, keepdim=True) - x_norm * (dloss_dx * x_norm).mean(-1, keepdim=True) * n_adj
    dloss_dx *= x_rstd
    return dloss_dx

def t_layernorm_bkwd2_x(dloss_dx, layer_params, x):
    x_2d = x.reshape((-1, x.shape[-1]))
    # TODO XXX XXX: investigate the difference in memory consumption between two
    #return normalized_x_bkwd2(dloss_dx * layer_params[0], x_2d)
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    return normalized_x_bkwd2_plus(dloss_dx_2d * layer_params[0], x_2d).reshape(dloss_dx.shape)

res15 = t_layernorm_bkwd2_x(dloss_dx, layer_params, aa)
print("RESULT:", res15.shape, res15[-2:, -4:, -8:])


print(f'\n### TRITON')
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
                    #num_stages: tl.constexpr,
                    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    # Load shared params
    # TODO T: I think triton will load them once into shared memory -> confirm
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    param1 = tl.load(param1_ptr + offsets, mask=mask, other=0.0)  
        
    for row_idx in tl.range(row_start, n_rows, row_step): #, num_stages):
        dloss_dx_row_start_ptr = dloss_dx_ptr + row_idx * dloss_dx_stride
        dloss_dx = tl.load(dloss_dx_row_start_ptr + offsets, mask=mask, other=0.0)
        dloss_dx = dloss_dx * param1
        x_row_start_ptr = x_ptr + row_idx * x_row_stride    
        x = tl.load(x_row_start_ptr + offsets, mask=mask, other=0.0)
        
        # compute mean and std for x
        x_sum = tl.sum(x, axis=0)
        x_mu = x_sum/ n_cols
        x_minus_mu = tl.where(mask, x-x_mu, 0) #x - x_mu
        x_minus_mu2 = x_minus_mu * x_minus_mu
        x_minus_mu2_sum = tl.sum(x_minus_mu2, axis=0)
        x_sigma2 = x_minus_mu2_sum / (n_cols-1)
        x_sigma = tl.sqrt_rn(x_sigma2)
        print('x_mu', x_mu)
        print('x_sigma', x_sigma)        
        
        # normalize x
        x_norm = x_minus_mu/x_sigma    
        print('x_norm', x_norm)        
        
        # bkwd quantities
        dloss_dx_sum = tl.sum(dloss_dx, axis=0)
        dloss_dx_mu = dloss_dx_sum/n_cols
        print(f'dloss_dx_mu', dloss_dx_mu)
        dloss_dx_x_norm = dloss_dx * x_norm
        dloss_dx_x_norm_sum = tl.sum(dloss_dx_x_norm, axis=0)
        dloss_dx_x_norm_mu = dloss_dx_x_norm_sum/n_cols
        print(f'dloss_dx_x_norm_mu', dloss_dx_x_norm_mu)
        
        n_adj = n_cols/(n_cols-1) # adjust for estimated vs calculated sigma
        output = dloss_dx - dloss_dx_mu - x_norm * dloss_dx_x_norm_mu * n_adj
        output = output/x_sigma
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + offsets, output, mask=mask)
    
def t_layernorm_bkwd2_x_t(dloss_dx:torch.Tensor, layer_params: torch.Tensor, x: torch.Tensor):
    # TODO T: without this reshape, this func is 2times faster?
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    x_2d = x.reshape((-1, x.shape[-1])) 
    n_rows, n_cols = x_2d.shape
    BLOCK_SIZE = 8 #triton.next_power_of_2(n_cols) # TODO: Fix for BLOCK_SIZE!= n_cols
    print(f'XXX', BLOCK_SIZE, n_cols)
    output = torch.empty_like(x_2d)
    # TODO T: The below numbers were tuned for A10 by choosing num_warps=8
    num_warps = 8
    num_stages = 2
    num_programs = min(n_rows, 480) 
    t_layernorm_bkwd2_x_k[(num_programs,)](dloss_dx_2d, layer_params[0], x_2d, output, 
                                       dloss_dx_2d.stride(0), x_2d.stride(0), output.stride(0), n_rows, n_cols, 
                                       BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps) #, num_stages=num_stages)
    return output.reshape(dloss_dx.shape)

res2 = t_layernorm_bkwd2_x_t(dloss_dx, layer_params, aa)
print("RESULT:", res2.shape, res2[-2:, -4:, -8:])

print("\n\n")
atol, rtol=1e-1, 1e-3
def pre_allclose(input, other, name, atol, rtol):
    allclose = torch.abs(input - other) - atol - rtol * torch.abs(other)
    print(name, torch.max(allclose))
pre_allclose(res1[0], res2[0], 'dx', atol=atol, rtol=rtol)
assert torch.allclose(res1, res2, atol=atol, rtol=rtol), (res1.shape, res2.shape, res1[:2, :4, :8], res2[:2, :4, :8])

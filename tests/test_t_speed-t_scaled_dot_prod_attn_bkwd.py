import sys
sys.path.append('../')


import torch
from model_triton import t_scaled_dot_prod_attn_fwd, t_scaled_dot_prod_attn_bkwd3

BS, H, N, D = 8, 12, 512, 64

dloss_dx = torch.randn((BS, H, N, D), device="cuda")
qkv = torch.randn((BS, H, 3, N, D), device="cuda")
mask = torch.tril(torch.ones((N,N), dtype=torch.bool, device="cuda")).unsqueeze(0).expand(BS, N, N)
train=False #True
p_gen_aux = 42
N_RUNS = 1 #10

from model_triton import t_scaled_dot_prod_attn_fwd3
old_acts = t_scaled_dot_prod_attn_fwd3(qkv, mask, train, p_gen_aux)[1]

def fn_naive(dloss_dx, qkv):
    return t_scaled_dot_prod_attn_bkwd3(dloss_dx, old_acts, qkv,  mask, train, p_gen_aux)
fn_jit = torch.compile(fn_naive)
#burn it
fn_jit(dloss_dx, qkv) 

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_jit(dloss_dx, qkv)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'JIT total', total)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_naive(dloss_dx, qkv)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Naive total', total)


import triton
import triton.language as tl

from model_triton import t_scaled_dot_prod_attn_fwd3_t, t_scaled_dot_prod_attn_bkwd3_t
output, new_acts = t_scaled_dot_prod_attn_fwd3_t(qkv, mask, train, p_gen_aux) # note new_acts contains output 
def fn_t(dloss_dx, qkv):
    return t_scaled_dot_prod_attn_bkwd3_t(dloss_dx, new_acts, qkv, mask, train, p_gen_aux)

# Burn it 
fn_t(dloss_dx, qkv)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_t(dloss_dx, qkv)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Triton total', total)


print(f'\ntriton.testing.Benchmark')
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BS'],  # Argument names to use as an x-axis for the plot.
        x_vals=[BS],
        #x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch', 'naive'],  # Possible values for `line_arg`.
        line_names=['Triton', 'torch', 'naive'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='t_scaled_dot_prod_attn_fwd_t',  # Name for the plot. Used also as a file name for saving the plot.
        args={'H':12, 'N':512, 'D':64},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(BS, H, N, D, provider):
    dloss_dx = torch.randn((BS, H, N, D), device="cuda")
    qkv = torch.randn((BS, H, 3, N, D), device="cuda")  
    
    old_acts = t_scaled_dot_prod_attn_fwd3(qkv, mask, train, p_gen_aux)[1]
    def fn_naive(dloss_dx, qkv):
        return t_scaled_dot_prod_attn_bkwd3(dloss_dx, old_acts, qkv,  mask, train, p_gen_aux)
    fn_jit = torch.compile(fn_naive)
    output, new_acts = t_scaled_dot_prod_attn_fwd3_t(qkv, mask, train, p_gen_aux) # note new_acts contains output 
    def fn_t(dloss_dx, qkv):
        return t_scaled_dot_prod_attn_bkwd3_t(dloss_dx, new_acts, qkv, mask, train, p_gen_aux)

    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_jit(dloss_dx, qkv), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_t(dloss_dx, qkv), quantiles=quantiles)
    if provider == 'naive':
       ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_naive(dloss_dx, qkv), quantiles=quantiles)
    #perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3) # TODO XXX: investigate whether this is right. In the tutorial they operate on different dtype
    perf = lambda ms: ms
    return perf(ms), perf(max_ms), perf(min_ms)
benchmark.run(print_data=True, show_plots=False)


print(f'\nOther diagnostic')
# GPU INFO
import torch
print(torch.cuda.get_device_properties("cuda"))
from triton.runtime import driver
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
SIZE_SMEM = properties["max_shared_mem"]
NUM_REGS = properties["max_num_regs"]
WARP_SIZE = properties["warpSize"] # Not 64 as A100
print("GPU properties", properties)

# Optimal hypers for kernel
from model_triton import t_scaled_dot_prod_attn_bkwd3_k, DROPOUT_RATE
num_stages = 4 if SIZE_SMEM > 200000 else 2
num_warps = 8
print(f'num_stages', num_stages, 'num_warps', num_warps)

q, k, v = torch.unbind(qkv, dim=2) # BS x H x N x D
BS, H, N, D = q.shape

dloss_dx = dloss_dx.reshape(BS*H, N, D)
q = q.reshape(BS*H, N, D)
k = k.reshape(BS*H, N, D)
v = v.reshape(BS*H, N, D)
#temp_mask = mask[0] # Asumme mask being the same across rows. TODO XXX: make that assumption throughput the code
acts0, acts1, _ = new_acts
acts0 = acts0.reshape(BS*H, N)
acts1 = acts1.reshape(BS*H, N)    
temp_output = output.reshape(BS*H, N, D)
dloss_dq = torch.zeros_like(q)
dloss_dk = torch.zeros_like(k)
dloss_dv = torch.zeros_like(v)    

BLOCK_SIZE_Q_N = 32
BLOCK_SIZE_K_T_N = 32
BLOCK_SIZE_D = triton.next_power_of_2(D)

if not train:
    p_gen_aux = 0 # Need to mock some value for triton to compile the kernel without errors
k_t = torch.transpose(k, -2, -1)
kernel = t_scaled_dot_prod_attn_bkwd3_k.warmup(
    dloss_dx, q, k_t, v, temp_output, mask, acts0, acts1,
    dloss_dq, dloss_dk, dloss_dv,
    dloss_dx.stride(0), dloss_dx.stride(1), dloss_dx.stride(2),
    q.stride(0), q.stride(1), q.stride(2), k_t.stride(0), k_t.stride(1), k_t.stride(2), 
    v.stride(0), v.stride(1), v.stride(2),
    temp_output.stride(0), temp_output.stride(1), temp_output.stride(2),        
    mask.stride(0), mask.stride(1), mask.stride(2),
    acts0.stride(0), acts1.stride(0),
    dloss_dq.stride(0), dloss_dq.stride(1), dloss_dq.stride(2),        
    dloss_dk.stride(0), dloss_dk.stride(1), dloss_dk.stride(2),
    dloss_dv.stride(0), dloss_dv.stride(1), dloss_dv.stride(2),
    train, DROPOUT_RATE, p_gen_aux,
    BS*H, H, N, D,
    BLOCK_SIZE_Q_N=BLOCK_SIZE_Q_N, BLOCK_SIZE_K_T_N = BLOCK_SIZE_K_T_N, BLOCK_SIZE_D=BLOCK_SIZE_D,
    grid=(1, ), num_warps=num_warps, num_stages=num_stages)

kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared
print(f'n_regs', n_regs, 'size_smem', size_smem)

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
print(f'occupancy', occupancy, SIZE_SMEM // size_smem)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
print(f'num_programs', num_programs)
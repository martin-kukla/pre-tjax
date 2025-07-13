import sys
sys.path.append('../')

import torch
from model_triton import t_scaled_dot_prod_attn_fwd3

BS, H, N, D = 8, 12, 512, 64
qkv = torch.randn((BS, H, 3, N, D), device="cuda")
mask = torch.tril(torch.ones((N,N), dtype=torch.bool, device="cuda")).unsqueeze(0).expand(BS, N, N)
train=False #True
p_gen_aux = 42
N_RUNS = 1 #10

def fn_naive(qkv):
    return t_scaled_dot_prod_attn_fwd3(qkv, mask, train, p_gen_aux)
fn_jit = torch.compile(fn_naive)
# burn it
_ = fn_jit(qkv) 

import time
t0 = time.time()
for _ in range(N_RUNS):
    result, _ = fn_jit(qkv)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'JIT total', total)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result, _ = fn_naive(qkv)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Naive total', total)


import triton
import triton.language as tl

from model_triton import t_scaled_dot_prod_attn_fwd3_t

def fn_t(qkv):
    return t_scaled_dot_prod_attn_fwd3_t(qkv, mask, train, p_gen_aux)
_ = fn_t(qkv) 

import time
t0 = time.time()
for _ in range(N_RUNS):
    result2, new_acts = fn_t(qkv)
    
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Triton total', total)

print(f'\ntriton.testing.Benchmark')
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BS'],  # Argument names to use as an x-axis for the plot.
        x_vals=[BS],
        #x_vals=[128 * 2**i for i in range(0, 6)],
        #x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`.
        #x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch', 'naive'],  # Possible values for `line_arg`.
        line_names=['Triton', 'torch', 'naive'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='t_scaled_dot_prod_attn_fwd_t',  # Name for the plot. Used also as a file name for saving the plot.
        args={'H':H, 'N':N, 'D':D},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(BS, H, N, D, provider):
    qkv = torch.randn((BS, H, 3, N, D), device="cuda")  
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_jit(qkv), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_t(qkv), quantiles=quantiles)
    if provider == 'naive':
       ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_naive(qkv), quantiles=quantiles)
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
from model_triton import t_scaled_dot_prod_attn_fwd3_k, DROPOUT_RATE
num_stages = 4 if SIZE_SMEM > 200000 else 2
num_warps = 8
print(f'num_stages', num_stages, 'num_warps', num_warps)

q, k, v = torch.unbind(qkv, dim=2) # BS x H x N x D
BS, H, N, D = q.shape
q = q.reshape(BS*H, N, D)
k = k.reshape(BS*H, N, D)
v = v.reshape(BS*H, N, D)
output = torch.zeros_like(q)
acts0 = torch.empty((BS*H, N), device=q.device)
acts1 = torch.empty_like(acts0)
BLOCK_SIZE_Q_N = 128
BLOCK_SIZE_K_T_N = 64
BLOCK_SIZE_D = triton.next_power_of_2(D) #16

k_t = torch.transpose(k, -2, -1)
kernel = t_scaled_dot_prod_attn_fwd3_k.warmup(
    q, k_t, v, mask, output, acts0, acts1,
    q.stride(0), q.stride(1), q.stride(2), k_t.stride(0), k_t.stride(1), k_t.stride(2), 
    v.stride(0), v.stride(1), v.stride(2),
    mask.stride(0), mask.stride(1), mask.stride(2), 
    output.stride(0), output.stride(1), output.stride(2),
    acts0.stride(0), acts1.stride(1),
    train, DROPOUT_RATE, p_gen_aux,
    BS*H, H, N, D,
    BLOCK_SIZE_Q_N = BLOCK_SIZE_Q_N, BLOCK_SIZE_K_T_N = BLOCK_SIZE_K_T_N, BLOCK_SIZE_D=BLOCK_SIZE_D, 
    Q_N_BLCKS = triton.cdiv(N, BLOCK_SIZE_Q_N),
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
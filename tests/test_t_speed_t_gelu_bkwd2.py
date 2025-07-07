import sys
sys.path.append('../')

import torch
from model_triton import t_gelu_bkwd2

BS, N, FFN = 8, 512, 3072
dloss_dx = torch.randn((BS, N, FFN), device="cuda")
aa = torch.randn((BS, N, FFN), device="cuda")
N_RUNS = 100

fn_naive = t_gelu_bkwd2
fn_jit = torch.compile(fn_naive)
fn_jit(dloss_dx, aa) # burn it

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_naive(dloss_dx, aa)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Naive total', total)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_jit(dloss_dx, aa)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'JIT total', total)

from model_triton import t_gelu_bkwd2_t
fn_t = t_gelu_bkwd2_t
fn_t(dloss_dx, aa) # burn it

import time
t0 = time.time()
for _ in range(N):
    result = fn_t(dloss_dx, aa)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Triton total', total)

print(f'\nOther diagnostic')
import triton
import triton.language as tl
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BS'],  # Argument names to use as an x-axis for the plot.
        x_vals=[8],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch', 'naive'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch', 'naive'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='t_gelu_fwd',  # Name for the plot. Used also as a file name for saving the plot.
        args={'N':512, 'FFN':3072},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(BS, N, FFN, provider):
    dloss_dx = torch.rand((BS, N, FFN), device="cuda", dtype=torch.float32)    
    x = torch.rand((BS, N, FFN), device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_jit(dloss_dx, x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_t(dloss_dx, x), quantiles=quantiles)
    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_naive(dloss_dx, x), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
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
from model_triton import t_gelu_bkwd2_k
num_stages = 4 if SIZE_SMEM > 200000 else 2
num_warps = 8
print(f'num_stages', num_stages, 'num_warps', num_warps)
dloss_dx_1d = dloss_dx.view(-1)
aa_1d = aa.view(-1)  
output = torch.empty_like(aa_1d)
n_elements = output.numel()


kernel = t_gelu_bkwd2_k.warmup(dloss_dx, aa_1d, output, n_elements, BLOCK_SIZE=1024, grid=(triton.cdiv(n_elements, 1024),))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared
print(f'n_regs', n_regs, 'size_smem', size_smem)

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
size_smem = max(1, size_smem)
print(f'occupancy', occupancy, SIZE_SMEM // size_smem)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
print(f'num_programs', num_programs)
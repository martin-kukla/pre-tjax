import sys
sys.path.append('../')

import torch
from model_triton import t_log_softmax_fwd, t_log_softmax_fwd_t

#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
aa = torch.randn((8, 12, 512, 512), device="cuda")
#aa = torch.randn((4096, 35374), device="cuda")
#aa = aa.view(-1)
N_RUNS = 10

fn_naive = t_log_softmax_fwd
fn_jit = torch.compile(fn_naive)
# burn it
fn_jit(aa) 

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_jit(aa)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'JIT total', total)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_naive(aa)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Naive total', total)

fn_t = t_log_softmax_fwd_t
# burn it
fn_t(aa)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_t(aa)
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
        #x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch', 'naive'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch', 'naive'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='t_log_softmax_fwd',  # Name for the plot. Used also as a file name for saving the plot.
        args={'H':12, 'N':512},  # Values for function arguments not in `x_names` and `y_name`.
        # TODO T: Use real M i.e. 
    ))
def benchmark(BS, H, N, provider):
    #dloss_dx = torch.rand(size, device="cuda", dtype=torch.float32)    
    x = torch.rand(BS, H, N, N, device="cuda", dtype=torch.float32)
    stream = getattr(torch, "cuda").Stream() # TODO XXX XXX: what is this stream about?
    getattr(torch, "cuda").set_stream(stream)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_jit(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_t(x), quantiles=quantiles)        
    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn_naive(x), quantiles=quantiles)
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
from model_triton import t_log_softmax_fwd_k
num_stages = 4 if SIZE_SMEM > 200000 else 2
num_warps = 8
x_2d = aa.reshape((-1, aa.shape[-1])) # TODO T: without this reshape, this func is 2times faster
n_rows, n_cols = x_2d.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols) 
output = torch.empty_like(x_2d)

kernel = t_log_softmax_fwd_k.warmup(x_2d, output, x_2d.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared
print(f'n_regs', n_regs, 'size_smem', size_smem)

occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
print(f'occupancy', occupancy, SIZE_SMEM // size_smem)
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy
print(f'num_programs', num_programs)
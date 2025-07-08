import sys
sys.path.append('../')

import torch
from loss_and_optimizer_triton import t_avg_cross_entropy_loss_bkwd3, t_avg_cross_entropy_loss_bkwd3_t

BS, N, V = 8, 512, 35374
y_labels = torch.randint(V, (BS, N), device="cuda")
x_logits = torch.randn((BS, N, V), device="cuda")

fn_naive = t_avg_cross_entropy_loss_bkwd3
fn_jit = torch.compile(fn_naive)
# burn it
fn_jit(y_labels, x_logits) 
N_RUNS=10 #5 #100

res1 = fn_naive(y_labels, x_logits)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_jit(y_labels, x_logits)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'JIT total', total)

import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_naive(y_labels, x_logits)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Naive total', total)

fn_t = t_avg_cross_entropy_loss_bkwd3_t

res2 = fn_t(y_labels, x_logits)
import time
t0 = time.time()
for _ in range(N_RUNS):
    result = fn_t(y_labels, x_logits)
torch.cuda.synchronize()
t1 = time.time()
total = t1-t0
print(f'Triton total', total)

print(f'\nOther diagnostic')
import torch
print(torch.cuda.get_device_properties("cuda"))
from triton.runtime import driver
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
SIZE_SMEM = properties["max_shared_mem"]
NUM_REGS = properties["max_num_regs"]
WARP_SIZE = properties["warpSize"] # Not 64 as A100
print(properties)

# Optimal hypers for kernel
from loss_and_optimizer_triton import t_avg_cross_entropy_loss_bkwd3_k
num_stages = 4 if SIZE_SMEM > 200000 else 2
num_warps = 8
y_labels_1d = y_labels.reshape((-1,))
y_labels_1d = y_labels_1d.to(torch.int64)
x_logits_2d = x_logits.reshape((y_labels.numel(), -1))
nonzero_count = torch.count_nonzero(y_labels_1d)

n_rows, n_cols = x_logits_2d.shape
BLOCK_SIZE = 1024 #512
loss = torch.zeros((1), device=x_logits_2d.device) # can we just return value from triton kernel instead? I doubt that
dloss_dx = torch.zeros_like(x_logits_2d)
aux_idx = torch.zeros((n_rows, BLOCK_SIZE), device=x_logits_2d.device, dtype=torch.bool)
aux_idx.scatter_(1, (y_labels_1d % BLOCK_SIZE).unsqueeze(1), True)
print(f'BLOCK_SIZE', BLOCK_SIZE, 'num_stages', num_stages, 'num_warps', num_warps)

kernel = t_avg_cross_entropy_loss_bkwd3_k.warmup(y_labels_1d, x_logits_2d, loss, dloss_dx, aux_idx, x_logits_2d.stride(0), dloss_dx.stride(0), aux_idx.stride(0), nonzero_count, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
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
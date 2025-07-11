import sys
sys.path.append('../')

import torch

BS, H, N, D = 8, 12, 512, 64
qkv = torch.randn((BS, H, 3, N, D), device="cuda")
mask = torch.tril(torch.ones((N,N), dtype=torch.bool, device="cuda")).unsqueeze(0).expand(BS, N, N)
train=False #True
p_gen_aux = 42
N_RUNS = 1


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

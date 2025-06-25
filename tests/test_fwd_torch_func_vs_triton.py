#### torch_func's fwd vs triton's fwd on train=False (we can't do this testing for train=True due to different RNGs tooling)
import sys
sys.path.append('../')

import torch
torch.set_printoptions(precision=6)
from torch.func import jacrev
from model_torch_func import init_transformer_gpt2
BS, H, N, D = 1, 2, 512, 64 # 1, 1, 3, 4 #2, 2, 5, 4
vocab_size = 128
layers = 2
p_gen_aux = [42] + [43,44,45] * layers
layers_params = init_transformer_gpt2(vocab_size, D, layers, H, 4*D, N)
y= torch.randint(vocab_size, (BS, N+1), device="cuda").to(torch.int32)
mask = torch.ones((BS, N, N), dtype=torch.bool, device="cuda")
train = False

for i, i_mask in enumerate(mask):
    mask[i] = torch.tril(i_mask)
    #mask[i] = torch.zeros_like(i_mask)

# itroduce aritifical paddign to break current code
k=10
mask[0, 1:, :]=0
print(mask)

lens = [N]*BS
import numpy as np
y_indices = torch.tensor(np.vstack([np.arange(el_len) for el_len in lens]), device="cuda")
print(f'y_indices', y_indices.shape)

from model_torch_func import batched_forward_gpt2
from model_triton import t_gpt2_forward_with_acts_t
y_in = y[:, :-1]
y_out = y[:, 1:]
y_indices = y_indices[:, 1:]
    
res2 = batched_forward_gpt2(layers_params, y_in, mask, y_indices, train)
print(res2.shape, res2)
#print(res2[1])
#print_res_shapes(res2[0]) 

print(f'----X----')

res3, acts3 = t_gpt2_forward_with_acts_t(layers_params, y_in, mask, y_indices, train, p_gen_aux)
print(res3.shape, res3)
#print_res_shapes(res3[0]) 

#assert torch.allclose(res2[0], res3[0], rtol=1e-2, atol=5e-3)
assert torch.allclose(res2, res3, rtol=1e-2, atol=5e-3) 
#assert torch.allclose(res2, res3, rtol=1e-3, atol=2e-4)

# print(f'----XXX----')

# def recursive_assert(a, b):
#     if isinstance(a, torch.Tensor):
#         assert isinstance(b, torch.Tensor)
#         torch.allclose(a, b, rtol=1e-3, atol=1e-4)
#     else:
#         assert not isinstance(b, torch.Tensor)
#         if len(a) != len(b):
#             return
#         assert len(a) == len(b), f'len(a) {len(a)}, len(b) {len(b)}'
#         for i, (ai, bi) in enumerate(zip(a, b)):
#             recursive_assert(ai, bi)
# recursive_assert(acts2, acts3)
# print(f'----XXX----')
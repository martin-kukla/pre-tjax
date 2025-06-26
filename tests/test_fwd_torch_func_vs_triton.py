#### torch_func's fwd vs triton's fwd on train=False (we can't do this testing for train=True due to different RNGs tooling)
import sys
sys.path.append('../')

import torch
torch.set_printoptions(precision=6)
from torch.func import jacrev
from model_torch_func import init_transformer_gpt2
BS, H, N, D = 2, 2, 512, 64 # 1, 1, 3, 4 #2, 2, 5, 4
vocab_size = 128
layers = 2
p_gen_aux = [42] + [43,44,45] * layers
layers_params = init_transformer_gpt2(vocab_size, D, layers, H, 4*D, N)
y= torch.randint(vocab_size, (BS, N+1), device="cuda").to(torch.int32)
mask = torch.ones((BS, N, N), dtype=torch.bool, device="cuda")
for i, i_mask in enumerate(mask):
    mask[i] = torch.tril(i_mask)
train = False

lens = [N]*BS
import numpy as np
y_indices = torch.tensor(np.vstack([np.arange(el_len) for el_len in lens]), device="cuda")

from model_torch_func import batched_forward_gpt2
from model_triton import t_gpt2_forward_with_acts_t
y_in = y[:, :-1]
y_out = y[:, 1:]
y_indices = y_indices[:, 1:]

# introduce aritifical padding to break current code
k=10
mask[1, k+1:, :]=0
print(mask)
    
res2 = batched_forward_gpt2(layers_params, y_in, mask, y_indices, train)
print(res2.shape, res2)

print(f'----X----')

res3, acts3 = t_gpt2_forward_with_acts_t(layers_params, y_in, mask, y_indices, train, p_gen_aux)
print(res3.shape, res3)

# TODO: this should be failing, as we compare on all logits from all positions right now (instead of non-padded ones)
# Due to assumption about causal masking in my FlashAttn impl, the result should be incorrect:
# For rows of the mask which are zeros (empty), we should be taking the average over all Vs, but we only take the ones in the lower triangular
assert torch.allclose(res2, res3, rtol=1e-2, atol=5e-3) 
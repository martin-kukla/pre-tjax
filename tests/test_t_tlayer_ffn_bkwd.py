import sys
sys.path.append('../')

import torch
from model_torch_func import tlayer_ffn_fwd, gelu, INIT_SCALE
from model_triton import t_tlayer_ffn_bkwd2_t, t_gelu_fwd

BS, N, D = 8, 512, 768
FFN = 4 * D
aa = torch.randn((BS, N, D), device="cuda")
p0 = INIT_SCALE*torch.randn((FFN, D), device="cuda")
p1 = torch.zeros((FFN,), device="cuda") #INIT_SCALE*torch.randn((FFN,), device="cuda") # zeros is real initialization, but can't capture some bugs..
p2 = INIT_SCALE*torch.randn((D, FFN), device="cuda")
p3 = torch.zeros((D,), device="cuda") #INIT_SCALE*torch.randn((D,), device="cuda") # zeros is real initialization, but can't capture some bugs..
params = (p0, p1, p2, p3)
dloss_dx = torch.randn((BS, N, D), device="cuda")

from functools import partial
(_, vjpfunc) = torch.func.vjp(partial(tlayer_ffn_fwd, activation_fn=t_gelu_fwd), params, aa)
res1 = vjpfunc(dloss_dx)
res1 = (res1[1], res1[0]) # flip order
print(res1[0].shape, len(res1[1]), res1[1][0].shape, res1[1][1].shape, res1[1][2].shape, res1[1][3].shape)
#print(res1[0][-2:, -4:, -8:])
#print(res1[1][-1][-32:])
#print(res1[1][-2][-4:, -16:])
#print(res1[1][-3][-32:])
# print(res1[1][-4][-4:, -16:])

res2 = t_tlayer_ffn_bkwd2_t(dloss_dx, params, aa, t_gelu_fwd)
print(res2[0].shape, len(res2[1]), res2[1][0].shape, res2[1][1].shape, res2[1][2].shape, res2[1][3].shape)
#print(res2[0][-2:, -4:, -8:])
#print(res2[1][-1][-32:])
#print(res2[1][-2][-4:, -16:])
#print(res2[1][-3][-32:])
# print(res2[1][-4][-4:, -16:])

# TODO: expand this method to account for atol, not just rtol. As the numbers are likely off right now
def pre_allclose(input, other, name, atol=5e-3, rtol=1e-7):
    allclose = torch.abs(input - other) - atol - rtol * torch.abs(other)
    print(name, torch.max(allclose))

pre_allclose(res1[0], res2[0], "dx")
assert torch.allclose(res1[0], res2[0], atol=5e-3, rtol=1e-7), (res1[0].shape, res2[0].shape, res1[0][-2:, -4:, -8:], res2[0][-2:, -4:, -8:])
#assert torch.allclose(res1, res2, atol=1.7e-2, rtol=0), (res1.shape, res2.shape, res1[-4:, -4:], res2[-4:, -4:])

pre_allclose(res1[1][-1], res2[1][-1], "dp -1")
assert torch.allclose(res1[1][-1], res2[1][-1], atol=1e-3, rtol=0), (res1[1][-1].shape, res2[1][-1].shape, res1[1][-1][-32:], res2[1][-1][-32:])

pre_allclose(res1[1][-2], res2[1][-2], "dp -2")
assert torch.allclose(res1[1][-2], res2[1][-2], atol=1e-3, rtol=0), (res1[1][-2].shape, res2[1][-2].shape, res1[1][-2][-4:, -16:], res2[1][-2][-4:,-16:])

pre_allclose(res1[1][-3], res2[1][-3], "dp -3")
assert torch.allclose(res1[1][-3], res2[1][-3], atol=1e-3, rtol=1e-7), (res1[1][-3].shape, res2[1][-3].shape, res1[1][-3][-32:], res2[1][-3][-32:])

pre_allclose(res1[1][-4], res2[1][-4], "dp -4")
assert torch.allclose(res1[1][-4], res2[1][-4], atol=5e-3, rtol=1e-7), (res1[1][-4].shape, res2[1][-4].shape, res1[1][-4][-4:, -16:], res2[1][-4][-4:,-16:])

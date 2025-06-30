import sys
sys.path.append('../')

import torch
from model_torch_func import tlayer_ffn_fwd, gelu
from model_triton import t_tlayer_ffn_bkwd2_t, t_gelu_fwd

BS, N, D = 8, 512, 768
FFN = 4 * D
aa = torch.randn((BS, N, D), device="cuda")
p0 = torch.randn((FFN, D), device="cuda")
p1 = torch.randn((FFN,), device="cuda")
p2 = torch.randn((D, FFN), device="cuda")
p3 = torch.randn((D,), device="cuda")
params = (p0, p1, p2, p3)
dloss_dx = torch.randn((BS, N, D), device="cuda")

from functools import partial
(_, vjpfunc) = torch.func.vjp(partial(tlayer_ffn_fwd, activation_fn=t_gelu_fwd), params, aa)
res1 = vjpfunc(dloss_dx)
res1 = (res1[1], res1[0]) # flip order
print(res1[0].shape, len(res1[1]), res1[1][0].shape, res1[1][1].shape, res1[1][2].shape, res1[1][3].shape)
print(res1[0][-2:, -4:, -8:])

res2 = t_tlayer_ffn_bkwd2_t(dloss_dx, params, aa, t_gelu_fwd)
print(res2[0].shape, len(res2[1]), res2[1][0].shape, res2[1][1].shape, res2[1][2].shape, res2[1][3].shape)
print(res2[0][-2:, -4:, -8:])

assert torch.allclose(res1[0], res2[0], atol=1e-2, rtol=5e-4), (res1[0].shape, res2[0].shape, res1[0][-2:, -4:, -8:], res2[0][-2:, -4:, -8:])
#assert torch.allclose(res1, res2, atol=1.7e-2, rtol=0), (res1.shape, res2.shape, res1[-4:, -4:], res2[-4:, -4:])

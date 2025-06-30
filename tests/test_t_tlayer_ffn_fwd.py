import sys
sys.path.append('../')

import torch
from model_torch_func import tlayer_ffn_fwd, gelu
from model_triton import t_tlayer_ffn_fwd_t, t_gelu_fwd

BS, N, D = 8, 512, 768
FFN = 4 * D
aa = torch.randn((BS, N, D), device="cuda")
p0 = torch.randn((FFN, D), device="cuda")
p1 = torch.randn((FFN,), device="cuda")
p2 = torch.randn((D, FFN), device="cuda")
p3 = torch.randn((D,), device="cuda")
params = (p0, p1, p2, p3)

res1 = tlayer_ffn_fwd(params, aa, gelu)
print(res1[-2:, -4:, -8:])


res2 = t_tlayer_ffn_fwd_t(params, aa, t_gelu_fwd)
print(res2[-2:, -4:, -8:])

assert torch.allclose(res1, res2, atol=1e-2, rtol=5e-4), (res1.shape, res2.shape, res1[-2:, -4:, -8:], res2[-2:, -4:, -8:])
#assert torch.allclose(res1, res2, atol=1.7e-2, rtol=0), (res1.shape, res2.shape, res1[-4:, -4:], res2[-4:, -4:])

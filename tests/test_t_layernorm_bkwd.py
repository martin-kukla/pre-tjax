import sys
sys.path.append('../')
import torch
from model_triton import t_layernorm_bkwd2_p, t_layernorm_bkwd2_p_t

dloss_dx = torch.randn((8, 512, 768), device="cuda")
#dloss_dx = torch.randn((1, 768), device="cuda")
#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
layer_params = (torch.randn((768), device="cuda"), torch.randn((768), device="cuda"))
#aa = torch.randn((2, 768), device="cuda")
aa = torch.randn((8, 512, 768), device="cuda")

res1 = t_layernorm_bkwd2_p(dloss_dx, layer_params, aa)
print(res1[0].shape, res1[1].shape)


res2 = t_layernorm_bkwd2_p_t(dloss_dx, layer_params, aa)
print(res1[0].shape, res2[1].shape)

atol, rtol=1e-1, 1e-2
diff_abs = torch.abs(res1[0] - res2[0])
allclose_bnd=atol+rtol*torch.abs(res2[0])
indices = (allclose_bnd<diff_abs).nonzero()
#print(allclose_bnd-diff_abs)
print(indices)
print(res1[0][indices])
print(res2[0][indices])


ind_diff_abs_max = torch.argmax(diff_abs)
print(torch.max(diff_abs), ind_diff_abs_max)
print(res1[0][ind_diff_abs_max], res2[0][ind_diff_abs_max])
assert torch.allclose(res1[0], res2[0], atol=atol, rtol=rtol), (res1[0].shape, res2[0].shape, res1[0][:100], res2[0][:100])
assert torch.allclose(res1[1], res2[1], atol=atol, rtol=rtol), (res1[1].shape, res2[1].shape, res1[1][:100], res2[1][:100])
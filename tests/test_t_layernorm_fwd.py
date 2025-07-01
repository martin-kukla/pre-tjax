import sys
sys.path.append('../')
import torch
from model_torch_func import layernorm_fwd
from model_triton import t_layernorm_fwd_t

layer_params = (torch.ones((768), device="cuda"), torch.zeros((768), device="cuda")) # real init
#layer_params = (torch.randn((768), device="cuda"), torch.randn((768), device="cuda")) # unreal init
aa = torch.randn((8, 512, 768), device="cuda")

res1=layernorm_fwd(layer_params, aa)
print(res1.shape, res1[-2:, -4:, -8:])

from model_triton import t_layernorm_fwd_t

res2 = t_layernorm_fwd_t(layer_params, aa)
print(res2.shape, res2[-2:, -4:, -8:])

atol, rtol=1e-3, 1e-5
def pre_allclose(input, other, name, atol, rtol):
    allclose = torch.abs(input - other) - atol - rtol * torch.abs(other)
    print(name, torch.max(allclose))
pre_allclose(res1, res2, 'x', atol, rtol)
assert torch.allclose(res1, res2, atol=1e-2, rtol=1e-3), (res1.shape, res2.shape, res1[0, -4:], res2[0, -4:])
import sys
sys.path.append('../')
import torch
from model_torch_func import layernorm_fwd
from model_triton import t_layernorm_bkwd2_p_t

dloss_dx = torch.randn((8, 512, 768), device="cuda")
layer_params = (torch.ones((768), device="cuda"), torch.zeros((768), device="cuda")) # real init
#layer_params = (torch.randn((768), device="cuda"), torch.randn((768), device="cuda")) # unreal init
aa = torch.randn((8, 512, 768), device="cuda")

(_, vjpfunc) = torch.func.vjp(layernorm_fwd, layer_params, aa)
res1 = vjpfunc(dloss_dx)[0]
print("res1")
print(res1[0].shape, res1[1].shape)
print(res1[0][-16:])
print(res1[1][-16:])


res2 = t_layernorm_bkwd2_p_t(dloss_dx, layer_params, aa)
print("res2")
print(res2[0].shape, res2[1].shape)
print(res2[0][-16:])
print(res2[1][-16:])

atol, rtol=1e-1, 1e-3
def pre_allclose(input, other, name, atol, rtol):
    allclose = torch.abs(input - other) - atol - rtol * torch.abs(other)
    print(name, torch.max(allclose))
    
pre_allclose(res1[0], res2[0], 'dp0', atol, rtol)
assert torch.allclose(res1[0], res2[0], atol=atol, rtol=rtol), (res1[0].shape, res2[0].shape, res1[0][:100], res2[0][:100])
pre_allclose(res1[1], res2[1], 'dp1', atol, rtol)
assert torch.allclose(res1[1], res2[1], atol=atol, rtol=rtol), (res1[1].shape, res2[1].shape, res1[1][:100], res2[1][:100])
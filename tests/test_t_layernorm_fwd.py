import sys
sys.path.append('../')
import torch
from model_triton import *

#dloss_dx = torch.randn((2, 512, 3072), device="cuda")
#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
layer_params = (torch.randn((768), device="cuda"), torch.randn((768), device="cuda"))
#aa = torch.randn((2, 768), device="cuda")
aa = torch.randn((8, 512, 768), device="cuda")
#aa = torch.randn((4096, 35374), device="cuda")
#aa = aa.view(-1)

res1=t_layernorm_fwd(layer_params, aa)
#result

from model_triton import t_layernorm_fwd_t

res2 = t_layernorm_fwd_t(layer_params, aa)
#result2

assert torch.allclose(res1, res2, atol=1e-2, rtol=1e-2), (res1.shape, res2.shape, res1[0, -4:], res2[0, -4:])
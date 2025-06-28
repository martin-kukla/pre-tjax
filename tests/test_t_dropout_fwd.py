import sys
sys.path.append('../')
import torch
from model_torch_func import dropout

#dloss_dx = torch.randn((2, 512, 3072), device="cuda")
#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
aa = torch.randn((8, 512, 768), device="cuda")
#aa = torch.randn((4096, 35374), device="cuda")
#aa = aa.view(-1
train=False #True

res1 = dropout(aa, train=train)
# res1

import triton
import triton.language as tl

from model_triton import t_dropout_fwd_t

res2=t_dropout_fwd_t(aa, train=train, p_gen_aux=42)
# res2

assert torch.allclose(res1, res2), (res1[0], res2[0])
print(f'res1', res1.shape, res1[0])
print(f'res2', res2.shape, res2[0])
import sys
sys.path.append('../')
import torch
from model_torch_func import dropout
from model_triton import t_dropout_bkwd2_t

dloss_dx = torch.randn((8, 512, 768), device="cuda")
#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
aa = torch.randn((8, 512, 768), device="cuda")
#aa = torch.randn((4096, 35374), device="cuda")
#aa = aa.view(-1)

from functools import partial
dropout_train = partial(dropout, train=True)
(_, vjpfunc) = torch.func.vjp(dropout_train, aa)
res1 = vjpfunc(dloss_dx)[0]
# print(res1[-2:, -4:, -8:])

res2 = t_dropout_bkwd2_t(dloss_dx, aa, train=True, p_gen_aux=42)
# print(res2[-2:, -4:, -8:])

assert torch.allclose(res1, res2), (res1[-2:, -4:, -8:], res2[-2:, -4:, -8:])
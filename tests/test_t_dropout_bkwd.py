import sys
sys.path.append('../')
import torch
from model_triton import t_dropout_bkwd2, t_dropout_bkwd2_t

dloss_dx = torch.randn((8, 512, 768), device="cuda")
#Two shapes are being used: [8, 12, 512, 512], and 4096, 35374
aa = torch.randn((8, 512, 768), device="cuda")
#aa = torch.randn((4096, 35374), device="cuda")
#aa = aa.view(-1)

res1= t_dropout_bkwd2(dloss_dx, aa, train=True, p_gen_aux=42) 
#print(res1)

res2 = t_dropout_bkwd2_t(dloss_dx, aa, train=True, p_gen_aux=42)
#print(res2)

assert torch.allclose(res1, res2), (res1[0], res2[0])
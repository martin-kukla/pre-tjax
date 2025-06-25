
import torch
from torch.func import jacrev
from model_torch_func import init_tlayer_gpt2
from model_triton import *

# BACKWARD PASS: compare implemented jacobian vs jacrev's

# Asssert equality of two nested lists of Jacobians
def assert_jacs(j_ll_l, j_ll_r, rtol=1e-01): # j_ll_l - jacobian list of lists left...
    assert len(j_ll_l) == len(j_ll_r)
    for j_l_l, j_l_r in zip(j_ll_l, j_ll_r):
        assert len(j_l_l) == len(j_l_r)
        for j_l, j_r in zip(j_l_l, j_l_r):
            #assert j_l.shape == j_r.shape
            assert torch.allclose(j_l, j_r, rtol)
    
def init_model_data():
    BS, H, N, D = 1, 2, 2, 4
    vocab_size = 6
    layers = 2
    layers_params = init_transformer_gpt2(vocab_size, D, layers, H, 4*D, N)
    y= torch.randint(vocab_size, (BS, N), device="cuda")
    mask = torch.ones((BS, N, N), dtype=torch.bool, device="cuda")
    return layers_params, y, mask
    
def test_full_bkwd1(): # eval, no mask
    layers_params, y, mask = init_model_data()

    from functools import partial
    fn = partial(t_gpt2_tlayers_fwd, mask=mask, indices=None, train=False)
    res = jacrev(fn)(layers_params, y)
    res2 = t_gpt2_tlayers_bkwd_p(layers_params, y, mask, None, train=False)
    
    assert_jacs(res, res2, 1e-02)

def test_full_bkwd2(): # eval, triangular mask
    layers_params, y, mask = init_model_data()
    #for i, i_mask in enumerate(mask):
    #    mask[i] = torch.zeros_like(i_mask)
    #print(mask)

    from functools import partial
    fn = partial(t_gpt2_tlayers_fwd, mask=mask, indices=None, train=False)
    res = jacrev(fn)(layers_params, y)
    res2 = t_gpt2_tlayers_bkwd_p(layers_params, y, mask, None, train=False)
    
    assert_jacs(res, res2, 1e-01)
# WORK IN PROGRESS: currently, coding up Triton kernals

# Convention for function names:
# *_fwd: forward pass
# *_fwd3: forward pass which gathers all activations (for activation checkpointing)
# *_bkwd_p: backward pass which computes Jacobian with respect to parameters
# *_bkwd_x: backward pass which computes Jacobian with respect to input
# *_bkwd2: backward pass which computes VJPs with respect to input and parameters
# *_bkwd3: backward pass which computes VJPs with respect to input and parameters (+ activation checkpointing)
# (all backward passes are writen from first principle with exception of bkwd for BMM - TODO)

### PARAMS + MODEL
DROPOUT_RATE = 0.1 # TODO: move it out, and pass as paramteter

import math
import torch

### PARAMS: they are the same as for Torch.Func, so import

from model_torch_func import init_transformer_gpt2, count_num_params

### MODEL in TRITON

def t_log_softmax_fwd(x_logits): # compute log_softmax from logits over the last dimension
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0] # as it returns (maxs, indices)
    return x_logits - torch.logsumexp(x_logits, axis=-1, keepdims=True)

def t_log_softmax_bkwd(x_logits):
    indims = x_logits.shape
    x_logits = x_logits.reshape((-1, x_logits.shape[-1]))
    
    BS, N = x_logits.shape
    
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0]
    logsums = torch.logsumexp(x_logits, axis=-1, keepdims=True)
    exp_logsums = torch.exp(logsums).unsqueeze(2) # Q: is it going to be numerically stable?
    
    # TODO XXX: can I use expand for the below line?
    jac = torch.repeat_interleave(-torch.exp(x_logits), N, dim=0, output_size=x_logits.numel())
    jac = jac.reshape(BS, N, N)
    jac_eye = torch.eye(N, device=x_logits.device).unsqueeze(0).expand(BS, N, N)
    jac = (exp_logsums * jac_eye + jac) / exp_logsums
    return torch.block_diag(*jac.unbind(0)).reshape(indims+indims)

def t_log_softmax_bkwd2(dloss_dx, x_logits):
    indims = x_logits.shape
    x_logits = x_logits.reshape((-1, x_logits.shape[-1]))
    
    BS, N = x_logits.shape
    
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0]
    logsums = torch.logsumexp(x_logits, axis=-1, keepdims=True)
    exp_logsums = torch.exp(logsums) # Q: is it going to be numerically stable?

    # TODO XXX: Add comments on maths why we can do elementwise VJP here
    jac = -torch.exp(x_logits)/exp_logsums
    return dloss_dx + dloss_dx.sum(-1, keepdim=True)*jac.reshape(dloss_dx.shape)

#     jac_eye = torch.eye(N, device=x_logits.device).unsqueeze(0).expand(BS, N, N)
#     jac = jac_eye + jac

#     # Since it's only rowise dependency of outputs on inputs, we don't create full jacobian.
#     # Instead, we compute VJP in rowise fashion:
#     # jac_softmax = torch.block_diag(*jac.unbind(0)).reshape(indims+indims)
#     # dloss_dx = _vjp_in_2d(dloss_dx, jac_softmax)
#     dloss_dx = _vjp_in_2d_rowise(dloss_dx, jac)
    
#     return dloss_dx

def t_embed_fwd(layer_params, x): # input: 1 x
    return layer_params[0][x] * math.sqrt(layer_params[0].shape[1]) # since layer_params[0] is vocab_size x emb_dim

def t_embed_bkwd(layer_params, x): # input: 1 x
    x_1d = x.reshape(-1)
    
    emb_size = layer_params[0].shape[1]    
    fn_outdim = torch.numel(x) * emb_size
    fn_indim =  torch.numel(layer_params[0]) # jacobian with respect to params
    jac = torch.zeros(fn_outdim, fn_indim, device=x.device)
    
    indices = torch.tile(torch.arange(emb_size, device=x.device), (x.numel(), 1))
    indices = ((x_1d * emb_size).unsqueeze(1) + indices).reshape(-1, 1)
    jac.scatter_(1, indices, math.sqrt(emb_size))
    
    return (jac.reshape( x.shape + (emb_size, layer_params[0].shape[0], layer_params[0].shape[1])), )

def t_embed_bkwd2(dloss_dx, layer_params, x): # input: 1 x
    emb_size = layer_params[0].shape[1]
    return t_indexing_bkwd2(dloss_dx, layer_params, x, math.sqrt(emb_size))

# VJP for operation of indexing "layer_params[x]".
# Apply additional coef to Jacobian before multipliation
def t_indexing_bkwd2(dloss_dx, layer_params, x, coef=1):
    x_1d = x.reshape(-1)
    
    # Note, in order to save space, don't create full Jacobian.
    # The Full Jacobian would be BS x N x D x V x D (two last dims are params, and V is vocab size)
    # Instead, we only need information to which vabulary each position maps
    # i.e. Jacobian of shape: BS x N x V
    jac = torch.zeros(torch.numel(x), layer_params[0].shape[0], device=x.device)
    jac.scatter_(1, x_1d.unsqueeze(1).to(torch.int64), coef)
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    return (torch.matmul(dloss_dx_2d.t(), jac).t(), )

def t_relu_fwd(x):
    return torch.where(torch.le(x, 0), 0, x) # as inputs are broadcastable in where&le - follows pytorch's implementation

def t_relu_bkwd(x):
    return torch.where(torch.le(x, 0), 0, 1)

def t_gelu_fwd(x):
    k = math.sqrt(2/math.pi)
    return 0.5 * x * (1 + torch.tanh(k * (x + 0.044715 * torch.pow(x,3))))

def t_gelu_bkwd(x): # TODO XXX XXX: I think maths can be simplified here? 
    k = math.sqrt(2/math.pi)
    tanh_term = torch.tanh(k * (x + 0.044715 * torch.pow(x,3)))
    tanh_dx = (1 - torch.pow(tanh_term, 2)) * k * ( 1 + 3 * 0.044715 * torch.pow(x,2))
    
    return 0.5 * (1 + tanh_term) + 0.5 * x * tanh_dx

def t_gelu_bkwd2(dloss_dx, x):
    jac = t_gelu_bkwd(x)
    return dloss_dx * jac # note, this is elementwise op

def t_linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params[0], 0, 1)) + layer_params[1][None, :] # since layer_params[0] is output_dim x emb_dim, layer_params[1] is output_dim

def t_linear_bkwd_p(layer_params, x): # input: N x D
    outdim = layer_params[1].shape[0]

    jac1 = t_proj_bkwd_p(layer_params[0], x)
    jac2 = torch.eye(outdim, device=x.device).expand(x.shape[:-1] + (outdim, outdim))
    return jac1, jac2

def _vjp_in_2d(v, jac):
    outdim = jac.shape[len(v.shape):]
    # TODO: It's just vector times matrix, is there cleaner/more efficient way of doing this?
    res = torch.matmul(v.view((1, -1)), jac.reshape((v.numel(), -1)))
    return res.view(outdim)

# Do VJP row-wise. Useful when a row doesn't depend on other rows (saves space)
def _vjp_in_2d_rowise(dloss_dx, rowise_jac): # dloss_dx: ... x IN_DIM, rowise_jac: BS x IN_DIM x OUT_DIM
    outdim = dloss_dx.shape[:-1] + rowise_jac.shape[-1:]
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    dloss_dx = torch.matmul(dloss_dx_2d.unsqueeze(1), rowise_jac).squeeze(1)
    return dloss_dx.reshape(outdim)

def _vjps_in_2d(v, jacs): # TODO XXX: reshape v once for all to speed up computation?
    return [_vjp_in_2d(v, j) for j in jacs] 

def t_linear_bkwd2_p(dloss_dx, layer_params, x): # input: N x D
    outdim = layer_params[1].shape[0]
    dloss_dp0 = t_proj_bkwd2_p(dloss_dx, layer_params[0], x)
    dloss_dp1 = dloss_dx.view((-1, outdim)).sum(dim=0)

    return dloss_dp0, dloss_dp1

def t_linear_bkwd_x(layer_params, x): # input: N x D
    return t_proj_bkwd_x(layer_params[0], x)

def t_linear_bkwd2_x(dloss_dx, layer_params, x): # input: N x D
    return t_proj_bkwd2_x(dloss_dx, layer_params[0], x)

def t_proj_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params, -2, -1)) # since layer_params is ... x output_dim x emb_dim

# TODO XXX: Placebolder. Code up Jacobian for bmm
def t_proj_bkwd_p(layer_params, x): # input: seq_len x emb_dim
    from torch.func import jacrev
    return jacrev(t_proj_fwd)(layer_params, x)
    
def my_t_proj_bkwd_p(layer_params, x): # input: seq_len x emb_dim
    indims = x.shape
    x = x.reshape((-1, x.shape[-1]))
    
    BS, N = x.shape
    outdim = layer_params.shape[-2]

    jac = x.unsqueeze(1).expand(BS, outdim, N)
    jac = jac.unsqueeze(-2).expand(BS, outdim, outdim, N)
    
    aux = torch.eye(outdim, device=x.device).unsqueeze(-1).expand(outdim, outdim, N)
    aux = aux.unsqueeze(0).expand(BS, outdim, outdim, N)
    
    outdims = indims[:-1] + (outdim, )
    return (jac*aux).reshape(outdims + layer_params.shape)

def t_proj_bkwd2_p(dloss_dx, layer_params, x): # input: N x D
    # There are numerical differences between using torch.func's jacrev and vjpfunc, and my VJP
    # all of it is in the region of floating point errors
    # (res, vjpfunc) = torch.func.vjp(t_proj_fwd, layer_params, x)
    # return vjpfunc(dloss_dx)[0]
    # return _vjp_in_2d(dloss_dx, t_proj_bkwd_p(layer_params, x))
    # TODO XXX XXX: This is because we overload t_proj_fwd in few places. Clean up:
    # We could either do reshape or have separate funcs
    # Note we don't want to copy memory to keep perf low
    dim = len(dloss_dx.shape)
    eq_str = 'abc, abd -> cd' if dim==3 else ('bc, bd -> cd' if dim==2 else 'abcde, axcdf -> bcef')
    return torch.einsum(eq_str, dloss_dx, x)


# TODO XXX: Placebolder. Code up Jacobian for bmm
def t_proj_bkwd_x(layer_params, x): # input: seq_len x emb_dim
    from torch.func import jacrev
    return jacrev(t_proj_fwd, argnums=1)(layer_params, x)

def my_t_proj_bkwd_x(layer_params, x): # input: seq_len x emb_dim
    indims = x.shape
    x = x.reshape((-1, x.shape[-1]))
    
    BS, N = x.shape
    outdim = layer_params.shape[-2]
    jac = layer_params.unsqueeze(0).expand(BS, outdim, N)
    jac = jac.unsqueeze(-2).expand(BS, outdim, BS, N)
    
    aux = torch.eye(BS, device=x.device).unsqueeze(1).expand(BS, outdim, BS)
    aux = aux.unsqueeze(-1).expand(BS, outdim, BS, N)
    
    outdims = indims[:-1] + (outdim, )
    return (jac*aux).reshape(outdims + indims)

def t_proj_bkwd2_x(dloss_dx, layer_params, x):
    # TODO XXX XXX: check whether it affects numerical values as t_proj_bkwd2_p does or not
    (_, vjpfunc) = torch.func.vjp(t_proj_fwd, layer_params, x)
    return vjpfunc(dloss_dx)[1]
    #return _vjp_in_2d(dloss_dx, t_proj_bkwd_x(layer_params, x))

def t_softmax_attn_fwd(q, k, mask, train, p_gen_aux=None):
    D = q.shape[-1]
    attn = torch.matmul(q, torch.transpose(k, -2, -1))
    attn = attn / math.sqrt(D)
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    sa = torch.exp(t_log_softmax_fwd(attn))
    sa = t_dropout_fwd(sa, train, p_gen_aux)
    return sa

def t_softmax_attn_bkwd(q, k, mask, train, p_gen_aux=None):
    D = q.shape[-1]
    attn = torch.matmul(q, torch.transpose(k, -2, -1))
    attn = attn / math.sqrt(D)
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    # TODO XXX: would the below line cause numerical stabliity issues?
    sa = torch.exp(t_log_softmax_fwd(attn)) 

    jac_dropout = t_dropout_bkwd(sa, train, p_gen_aux)
    #TODO: Note, we are overloading _mult.., as right is not Jacobian...
    sa = _mult_jacs_in_2d(jac_dropout, [sa], sa)[0] 
    
    # TODO XXX: Clean up below..
    jac_sa_x = sa[..., None, None, None, None] * t_log_softmax_bkwd(attn)
    jac1 = torch.matmul(jac_sa_x, k/math.sqrt(D))
    jac2 = torch.matmul(q.transpose(-2,-1), jac_sa_x/math.sqrt(D)).transpose(-2,-1)
    # Account for mask:
    jac_mask = torch.unsqueeze(mask,dim=1)[..., None, None, None, None]
    jac1 = torch.where(jac_mask, jac1, 0)
    jac2 = torch.where(jac_mask, jac2, 0)
    return jac1, jac2

def t_softmax_attn_bkwd2(dloss_dx, q, k, mask, train, p_gen_aux=None):
    D = q.shape[-1]
    attn = torch.matmul(q, torch.transpose(k, -2, -1))
    attn = attn / math.sqrt(D)
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    # TODO XXX: would the below line cause numerical stabliity issues?
    sa = torch.exp(t_log_softmax_fwd(attn)) 

    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, sa, train, p_gen_aux)
    dloss_dx = dloss_dx * sa #note, sa acts as jac_exp (exp is element-wise op). TODO: check if this is correct?
    
    # TODO XXX XXX : torch.func's jacrev/vjp give 2 different results,
    # which, importantly, are different than my implementation.
    # It's all in teh region of floating points errors..
    #from torch.func import jacrev
    #qk_t_bmm_fn = lambda q, k: torch.matmul(q, k.transpose(-2, -1))/math.sqrt(D)
    #bmm_jac_k = jacrev(qk_t_bmm_fn, argnums=(1))(q, k)
    #(_, vjpfunc) = torch.func.vjp(qk_t_bmm_fn, q, k) 
    
    dloss_dx = t_log_softmax_bkwd2(dloss_dx, attn)
    dloss_dx = torch.where(torch.unsqueeze(mask,dim=1), dloss_dx, 0)
    dloss_dq = torch.matmul(dloss_dx, k/math.sqrt(D))
    #dloss_dk = _vjp_in_2d(dloss_dx, bmm_jac_k)
    #dloss_dk = vjpfunc(dloss_dx)[1] # note, this also computes [0]...
    dloss_dk = torch.einsum('abcd, abce->abde', dloss_dx/math.sqrt(D), q)
    
    return dloss_dq, dloss_dk

def t_scaled_dot_prod_attn_fwd(qkv, mask, train=True, p_gen_aux=None): # inputs: BS x H x 3 x N x D, mask: BS x N(q) x N(k)
    q, k, v = torch.unbind(qkv, dim=2) # BS x H x N x D
    softmaxed_attn = t_softmax_attn_fwd(q, k, mask, train, p_gen_aux)
    return torch.matmul(softmaxed_attn, v) # output: BS x N x D

def t_scaled_dot_prod_attn_fwd3(qkv, mask, train=True, p_gen_aux=None): # inputs: BS x H x 3 x N x D, mask: BS x N(q) x N(k)
    q, k, v = torch.unbind(qkv, dim=2) # BS x H x N x D
    softmaxed_attn = t_softmax_attn_fwd(q, k, mask, train, p_gen_aux)
    return torch.matmul(softmaxed_attn, v), [softmaxed_attn] # output: BS x N x D

def t_scaled_dot_prod_attn_bkwd(qkv, mask, train=True, p_gen_aux=None): # inputs: BS x H x 3 x N x D, mask: BS x N(q) x N(k)
    BS, H, _, N, D = qkv.shape
    q, k, v = torch.unbind(qkv, dim=2)
    
    sa = t_softmax_attn_fwd(q, k, mask, train, p_gen_aux)
    jac_sa_q, jac_sa_k = t_softmax_attn_bkwd(q, k, mask, train, p_gen_aux)     
    
    # TODO XXX: code up jacobian for bmm
    from torch.func import jacrev
    bbm_fn = lambda m1, m2: torch.matmul(m1, m2)
    jac_bmm_sa, jac_v = jacrev(bbm_fn, argnums=(0,1))(sa, v)
    
    jacs_q_k = _mult_jacs_in_2d(jac_bmm_sa, [jac_sa_q, jac_sa_k], sa)   
    
    return jacs_q_k[0], jacs_q_k[1], jac_v

def t_scaled_dot_prod_attn_bkwd2(dloss_dx, qkv, mask, train=True, p_gen_aux=None): # inputs: BS x H x 3 x N x D, mask: BS x N(q) x N(k)
    BS, H, _, N, D = qkv.shape
    q, k, v = torch.unbind(qkv, dim=2)
    sa = t_softmax_attn_fwd(q, k, mask, train, p_gen_aux)
    
    # propagate back
    # TODO XXX: code up jacobian for bmm
    from torch.func import jacrev
    bbm_fn = lambda m1, m2: torch.matmul(m1, m2)
    # TODO XXX XXX: Investigate why the numerical differences between jacrev and vjp
    # jac_bmm_sa, jac_v = jacrev(bbm_fn, argnums=(0,1))(sa, v)
    # dloss_dsa, dloss_dv = _vjps_in_2d(dloss_dx, [jac_bmm_sa, jac_v])
    (_, vjpfunc) = torch.func.vjp(bbm_fn, sa, v)
    dloss_dsa, dloss_dv = vjpfunc(dloss_dx)
    dloss_dq, dloss_dk = t_softmax_attn_bkwd2(dloss_dsa, q, k, mask, train, p_gen_aux)
    
    return dloss_dq, dloss_dk, dloss_dv

def t_scaled_dot_prod_attn_bkwd3(dloss_dx, acts, qkv, mask, train=True, p_gen_aux=None): # inputs: BS x H x 3 x N x D, mask: BS x N(q) x N(k)
    BS, H, _, N, D = qkv.shape
    q, k, v = torch.unbind(qkv, dim=2)
    sa = acts[0]
    
    # propagate back: bmm (i.e. sa * v)  + SA
    dloss_dsa = torch.einsum(f'abcd, abed -> abce', dloss_dx, v)
    dloss_dv = torch.einsum(f'abcd, abce -> abed', dloss_dx, sa)
    dloss_dq, dloss_dk = t_softmax_attn_bkwd2(dloss_dsa, q, k, mask, train, p_gen_aux)
    
    return dloss_dq, dloss_dk, dloss_dv

# TODO XXX: Remove below
# TODO XXX: Support for heads>1
# TODO XXX: replace mult with the generic newer _mult
def old_t_scaled_dot_prod_attn_bkwd(qkv, mask, train=True): # inputs: batch_size x heads x 3 x seq_len x emb_dim, mask: batch_size x seq_len(q) x seq_len(k)
    BS, H, _, N, D = qkv.shape
    q, k, v = torch.unbind(qkv, dim=2)
    
    softmaxed_attn = t_softmax_attn_fwd(q, k, mask, train)
    dsoftmaxed_attn_dq, dsoftmaxed_attn_dk = t_softmax_attn_bkwd(q, k, mask, train)  
    
    v_2d = v.reshape((-1, v.shape[-1]))
    def mult_with_v_2d_bkwd(A): # A being 8D here
        A_4d_outdim_shape = (A.shape[0] * A.shape[1] *A.shape[2], A.shape[3])
        A_4d = A.reshape(A_4d_outdim_shape+v_2d.shape)
        
        # TODO XXX: Clean up these reshapes
        jac_a = torch.matmul(v_2d.transpose(1, 0), A_4d.transpose(1, 0).reshape(v_2d.shape[0], -1))
        jac_a = jac_a.reshape(jac_a.shape[0], -1, v_2d.numel()).transpose(1, 0)
        return jac_a
    
    jac_q = mult_with_v_2d_bkwd(dsoftmaxed_attn_dq).reshape(v.shape + v.shape)
    jac_k = mult_with_v_2d_bkwd(dsoftmaxed_attn_dk).reshape(v.shape + v.shape)
    
    jac_v = softmaxed_attn
    # TODO XXX: Fix this very ugly iterative reshape 
    jac_v = jac_v.reshape((-1, ) + jac_v.shape[3:])
    res = []
    for it in jac_v:
        res.append(torch.block_diag( *[it.unsqueeze(1)]*v.shape[-1]))
    jac_v = torch.stack(res).reshape(v.shape + v.shape)
    
    return jac_q, jac_k, jac_v

def t_tlayer_attn_heads_fwd(layer_params, qkv, mask, train, p_gen_aux=None): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    qkv = torch.stack(qkv,dim=-3) # batch_size x 3 x seq_len x emb_dim
    
    proj_qkv = t_proj_fwd(layer_params, torch.unsqueeze(qkv, 1)) # batch_size x heads x 3 x seq_len x emb_dim
    return t_scaled_dot_prod_attn_fwd(proj_qkv, mask, train, p_gen_aux)

def t_tlayer_attn_heads_fwd3(layer_params, qkv, mask, train, p_gen_aux=None): # params: H x 3 x D/H x D, input: BS x N x D
    qkv = torch.stack(qkv,dim=-3) # batch_size x 3 x seq_len x emb_dim
    
    proj_qkv = t_proj_fwd(layer_params, torch.unsqueeze(qkv, 1)) # batch_size x heads x 3 x seq_len x emb_dim
    return t_scaled_dot_prod_attn_fwd3(proj_qkv, mask, train, p_gen_aux)

def t_tlayer_attn_heads_bkwd_p(layer_params, qkv, mask, train, p_gen_aux=None): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    
    proj_qkv = t_proj_fwd(layer_params, qkv)
    jac_proj_p = t_proj_bkwd_p(layer_params, qkv)
     
    jac_sdpa_x = t_scaled_dot_prod_attn_bkwd(proj_qkv, mask, train, p_gen_aux)
    jac_sdpa_x = torch.stack(jac_sdpa_x, dim=-3)
    jac_p = _mult_jacs_in_2d(jac_sdpa_x, [jac_proj_p], qkv)[0]
    return jac_p

def t_tlayer_attn_heads_bkwd2_p(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # params: H x 3 x D/H x D, input: BS x N x D
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    proj_qkv = t_proj_fwd(layer_params, qkv)
     
    # propagate back
    dloss_dx = t_scaled_dot_prod_attn_bkwd2(dloss_dx, proj_qkv, mask, train, p_gen_aux)
    dloss_dx = torch.stack(dloss_dx, dim=-3)
    dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params, qkv)
    return dloss_dp

def t_tlayer_attn_heads_bkwd_x(layer_params, qkv, mask, train, p_gen_aux=None): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    
    proj_qkv = t_proj_fwd(layer_params, qkv)
    jac_proj_x = t_proj_bkwd_x(layer_params, qkv)
     
    jac_sdpa_x = t_scaled_dot_prod_attn_bkwd(proj_qkv, mask, train, p_gen_aux)
    jac_sdpa_x = torch.stack(jac_sdpa_x, dim=-3)
    jac_x = _mult_jacs_in_2d(jac_sdpa_x, [jac_proj_x], qkv)[0]
    
    return jac_x.squeeze(-4).unbind(-3)

def t_tlayer_attn_heads_bkwd2_x(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # params: H x 3 x D/H x D, input: BS x S x D
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    proj_qkv = t_proj_fwd(layer_params, qkv)
    
    # propagate back
    dloss_dx = t_scaled_dot_prod_attn_bkwd2(dloss_dx, proj_qkv, mask, train, p_gen_aux)
    dloss_dx = torch.stack(dloss_dx, dim=-3)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params, qkv)
    dloss_dqkv = dloss_dx.squeeze(-4).unbind(-3)
    return dloss_dqkv

def t_tlayer_attn_heads_bkwd2(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # params: H x 3 x D/H x D, input: BS x S x D
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    proj_qkv = t_proj_fwd(layer_params, qkv)
    
    # propagate back
    dloss_dx = t_scaled_dot_prod_attn_bkwd2(dloss_dx, proj_qkv, mask, train, p_gen_aux)
    dloss_dx = torch.stack(dloss_dx, dim=-3)
    dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params, qkv)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params, qkv)
    dloss_dqkv = dloss_dx.squeeze(-4).unbind(-3)
    
    return dloss_dqkv, dloss_dp

def t_tlayer_attn_heads_bkwd3(dloss_dx, acts, layer_params, qkv, mask, train, p_gen_aux=None): # params: H x 3 x D/H x D, input: BS x S x D
    qkv = torch.stack(qkv,dim=-3).unsqueeze(1)
    proj_qkv = t_proj_fwd(layer_params, qkv)
    
    # propagate back
    dloss_dx = t_scaled_dot_prod_attn_bkwd3(dloss_dx, acts, proj_qkv, mask, train, p_gen_aux)
    dloss_dx = torch.stack(dloss_dx, dim=-3)
    dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params, qkv)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params, qkv)
    dloss_dqkv = dloss_dx.squeeze(-4).unbind(-3)
    
    return dloss_dqkv, dloss_dp

def t_tlayer_attn_fwd(layer_params, qkv, mask, train, p_gen_aux=None): # input: batch_size x seq_len x emb_dim
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    return t_proj_fwd(layer_params[-1], attn)

def t_tlayer_attn_fwd3(layer_params, qkv, mask, train, p_gen_aux=None): # input: batch_size x seq_len x emb_dim
    heads_attns, acts = t_tlayer_attn_heads_fwd3(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    return t_proj_fwd(layer_params[-1], attn), [acts, heads_attns]

def t_tlayer_attn_bkwd_p(layer_params, qkv, mask, train, p_gen_aux=None): # input: batch_size x seq_len x emb_dim
    jac_heads_attns_p = t_tlayer_attn_heads_bkwd_p(layer_params[0], qkv, mask, train, p_gen_aux)
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape  
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    jac_heads_attns_p = jac_heads_attns_p.transpose(1, 2).reshape((BS, N, -1) + layer_params[0].shape)  
    
    jac_proj_x = t_proj_bkwd_x(layer_params[-1], attn)
    jac_proj_p = t_proj_bkwd_p(layer_params[-1], attn)
    
    res = _mult_jacs_in_2d(jac_proj_x, [jac_heads_attns_p], qkv[0])[0]
    return res, jac_proj_p

def t_tlayer_attn_bkwd2_p(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # input: batch_size x seq_len x emb_dim
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape  
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D

    # propagate back
    proj_dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params[-1], attn)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params[-1], attn)
    dloss_dx = dloss_dx.reshape(BS, N, H, D).transpose(1, 2) # unflatten H+D, then swap back H and N
    heads_attns_dloss_dp = t_tlayer_attn_heads_bkwd2_p(dloss_dx, layer_params[0], qkv, mask, train, p_gen_aux)    
    
    return heads_attns_dloss_dp, proj_dloss_dp

def t_tlayer_attn_bkwd_x(layer_params, qkv, mask, train, p_gen_aux=None): # input: batch_size x seq_len x emb_dim
    jac_heads_attns_x = t_tlayer_attn_heads_bkwd_x(layer_params[0], qkv, mask, train, p_gen_aux)
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    jac_heads_attns_x = [j.transpose(1, 2).reshape((BS, N, -1) + qkv[0].shape) for j in jac_heads_attns_x]
    
    jac_proj_x = t_proj_bkwd_x(layer_params[-1], attn)
    return tuple(_mult_jacs_in_2d(jac_proj_x, jac_heads_attns_x, qkv[0]))

def t_tlayer_attn_bkwd2_x(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # input: BS x N x D
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    
    # propagate back
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params[-1], attn)
    dloss_dx = dloss_dx.reshape(BS, N, H, D).transpose(1, 2) # unflatten H+D, then swap back H and N
    dloss_dx = t_tlayer_attn_heads_bkwd2_x(dloss_dx, layer_params[0], qkv, mask, train, p_gen_aux)
    
    return dloss_dx

def t_tlayer_attn_bkwd2(dloss_dx, layer_params, qkv, mask, train, p_gen_aux=None): # input: BS x N x D
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train, p_gen_aux)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    
    # propagate back
    proj_dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params[-1], attn)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params[-1], attn)
    dloss_dx = dloss_dx.reshape(BS, N, H, D).transpose(1, 2) # unflatten H+D, then swap back H and N
    dloss_dx, heads_attns_dloss_dp = t_tlayer_attn_heads_bkwd2(dloss_dx, layer_params[0], qkv, mask, train, p_gen_aux)
    
    return dloss_dx, (heads_attns_dloss_dp, proj_dloss_dp)

def t_tlayer_attn_bkwd3(dloss_dx, acts, layer_params, qkv, mask, train, p_gen_aux=None): # input: BS x N x D
    heads_attns = acts[-1]
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(1, 2).reshape((BS, N, -1)) # Swap H and N, then flatten H+D
    
    # propagate back
    proj_dloss_dp = t_proj_bkwd2_p(dloss_dx, layer_params[-1], attn)
    dloss_dx = t_proj_bkwd2_x(dloss_dx, layer_params[-1], attn)
    dloss_dx = dloss_dx.reshape(BS, N, H, D).transpose(1, 2) # unflatten H+D, then swap back H and N
    dloss_dx, heads_attns_dloss_dp = t_tlayer_attn_heads_bkwd3(dloss_dx, acts[-2], layer_params[0], qkv, mask, train, p_gen_aux)
    
    return dloss_dx, (heads_attns_dloss_dp, proj_dloss_dp)

def t_tlayer_ffn_fwd(layer_params, x, activation_fn): # input: seq_len x emb_dim
    x = t_linear_fwd((layer_params[0], layer_params[1]), x)
    x = activation_fn(x)
    x = t_linear_fwd((layer_params[2], layer_params[3]), x)
    return x

def t_tlayer_ffn_fwd3(layer_params, x, activation_fn): # input: seq_len x emb_dim
    x = t_linear_fwd((layer_params[0], layer_params[1]), x)
    acts = [x]
    x = activation_fn(x)
    x = t_linear_fwd((layer_params[2], layer_params[3]), x)
    return x, acts

def t_tlayer_ffn_bkwd_p(layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    
    act_fn_bkwd = t_gelu_bkwd if activation_fn==t_gelu_fwd else t_relu_bkwd
    
    jac1 = t_linear_bkwd_p((layer_params[0], layer_params[1]), x_2d)
    x_2d = t_linear_fwd((layer_params[0], layer_params[1]), x_2d)
    dact_dx = act_fn_bkwd(x_2d)
    x_2d = activation_fn(x_2d)
    jac2 = t_linear_bkwd_p((layer_params[2], layer_params[3]), x_2d)
    dffn2_dx = t_linear_bkwd_x((layer_params[2], layer_params[3]), x_2d)
    dffn2_act_dx = dact_dx * dffn2_dx #Note dact_dx is only 2D, but torch will add other dims
    jac1 = (torch.einsum('abcd,cdef->abef', dffn2_act_dx, jac1[0]),
            torch.einsum('abcd,cdf->abf', dffn2_act_dx, jac1[1]))
    
    return [j.reshape(x.shape+p.shape) for j, p in zip(jac1+jac2, layer_params)]

def t_tlayer_ffn_bkwd2_p(dloss_dx, layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    # note, t_relu_bkwd2 is not implemented yet
    act_fn_bkwd2 = t_gelu_bkwd2 if activation_fn==t_gelu_fwd else t_relu_bkwd2
    
    x_2d_in0 = x_2d
    x_2d = t_linear_fwd((layer_params[0], layer_params[1]), x_2d)
    x_2d_in1 = x_2d
    x_2d = activation_fn(x_2d)
    
    # propagate back
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    ffn2_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = act_fn_bkwd2(dloss_dx_2d, x_2d_in1)
    ffn1_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)
    
    return tuple(ffn1_dloss_dp+ffn2_dloss_dp)

def t_tlayer_ffn_bkwd_x(layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    
    act_fn_bkwd = t_gelu_bkwd if activation_fn==t_gelu_fwd else t_relu_bkwd
    
    dffn1_dx = t_linear_bkwd_x((layer_params[0], layer_params[1]), x_2d)
    x_2d = t_linear_fwd((layer_params[0], layer_params[1]), x_2d)
    dact_dx = act_fn_bkwd(x_2d)
    x_2d = activation_fn(x_2d)
    dffn2_dx = t_linear_bkwd_x((layer_params[2], layer_params[3]), x_2d)
    dffn2_act_dx = dact_dx * dffn2_dx #Note dact_dx is only 2D, but torch will add other dims
    jac = torch.einsum('abcd,cdef->abef', dffn2_act_dx, dffn1_dx)
    return jac.reshape(x.shape+x.shape)

def t_tlayer_ffn_bkwd2_x(dloss_dx, layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    # note, t_relu_bkwd2 is not implemented yet
    act_fn_bkwd2 = t_gelu_bkwd2 if activation_fn==t_gelu_fwd else t_relu_bkwd2 
    
    x_2d_in0 = x_2d
    x_2d = t_linear_fwd((layer_params[0], layer_params[1]), x_2d)
    x_2d_in1 = x_2d
    x_2d = activation_fn(x_2d)
    
    # propagate back
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = act_fn_bkwd2(dloss_dx_2d, x_2d_in1)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)

    return dloss_dx_2d.reshape(x.shape)

def t_tlayer_ffn_bkwd2(dloss_dx, layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    # note, t_relu_bkwd2 is not implemented yet
    act_fn_bkwd2 = t_gelu_bkwd2 if activation_fn==t_gelu_fwd else t_relu_bkwd2 
    
    x_2d_in0 = x_2d
    x_2d = t_linear_fwd((layer_params[0], layer_params[1]), x_2d)
    x_2d_in1 = x_2d
    x_2d = activation_fn(x_2d)
    
    # propagate back
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    ffn2_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = act_fn_bkwd2(dloss_dx_2d, x_2d_in1)
    ffn1_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)

    return dloss_dx_2d.reshape(x.shape), tuple(ffn1_dloss_dp+ffn2_dloss_dp)

def t_tlayer_ffn_bkwd3(dloss_dx, acts, layer_params, x, activation_fn):
    x_2d = x.reshape((-1, x.shape[-1]))
    # note, t_relu_bkwd2 is not implemented yet
    act_fn_bkwd2 = t_gelu_bkwd2 if activation_fn==t_gelu_fwd else t_relu_bkwd2 
    
    x_2d_in0 = x_2d
    x_2d = acts[0].reshape((-1, acts[0].shape[-1]))  # TODO XXX XXX: align shapes of fwd and bkwd's fwd (perf sufers)
    x_2d_in1 = x_2d
    x_2d = activation_fn(x_2d)
    
    # propagate back
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    ffn2_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[2], layer_params[3]), x_2d)
    dloss_dx_2d = act_fn_bkwd2(dloss_dx_2d, x_2d_in1)
    ffn1_dloss_dp = t_linear_bkwd2_p(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)
    dloss_dx_2d = t_linear_bkwd2_x(dloss_dx_2d, (layer_params[0], layer_params[1]), x_2d_in0)

    return dloss_dx_2d.reshape(x.shape), tuple(ffn1_dloss_dp+ffn2_dloss_dp)

def t_dropout_fwd(x, train=True, p_gen_aux=None):
    if not train: # As we jit the whole loss/inference, the train param is known at tracing time.
        return x * (1-DROPOUT_RATE)
    
    assert p_gen_aux is not None
    generator = torch.Generator(device=x.device).manual_seed(p_gen_aux)
    mask = torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE), generator=generator)
    
    return x * mask

def t_dropout_bkwd(x, train=True, p_gen_aux=None):
    eyed_jac = torch.eye(x.numel(), device=x.device).reshape(x.shape + x.shape)
    if not train: # we will never use this jacobian..
        return eyed_jac * (1-DROPOUT_RATE)

    assert p_gen_aux is not None
    generator = torch.Generator(device=x.device).manual_seed(p_gen_aux)
    mask = torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE), generator=generator) 
    return eyed_jac * mask

def t_dropout_bkwd2(dloss_dx, x, train=True, p_gen_aux=None):
    if not train: # we will never use this jacobian..
        return dloss_dx * (1-DROPOUT_RATE)

    assert p_gen_aux is not None
    generator = torch.Generator(device=x.device).manual_seed(p_gen_aux)
    mask = torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE), generator=generator) 
    return dloss_dx * mask

def t_layernorm_fwd(layer_params, x):
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims=True) # TODO XXX: Compute variance, add epsilon and take a sqrt instead (in order to avoid division by zero)
    normalized_x = (x - x_mean) / x_std
    return torch.multiply(normalized_x, layer_params[0][None, :]) + layer_params[1][None, :] # since both layer_params are output_dim x

def t_layernorm_bkwd_p(layer_params, x):
    x_indims = x.shape
    N = x.shape[-1]
    outdim=layer_params[1].shape[0]
    
    
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims=True)
    jac1 = ((x-x_mean)/x_std).unsqueeze(-1).expand(x_indims + (N, ))
    jac1_aux = torch.eye(N, device=x.device) # just used for reshaping
    jac2 = torch.eye(outdim, device=x.device).expand(x_indims[:-1] + (outdim, outdim))
    return jac1 *jac1_aux, jac2

def t_layernorm_bkwd2_p(dloss_dx, layer_params, x):
    x_indims = x.shape
    N = x.shape[-1]
    outdim=layer_params[1].shape[0]
    
    
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims=True)
    jac1 = ((x-x_mean)/x_std).unsqueeze(-1).expand(x_indims + (N, ))
    jac1_aux = torch.eye(N, device=x.device) # just used for reshaping
    jac2 = torch.eye(outdim, device=x.device).expand(x_indims[:-1] + (outdim, outdim))
    
    return _vjp_in_2d(dloss_dx, jac1 *jac1_aux), _vjp_in_2d(dloss_dx, jac2)

def normalized_x_bkwd(x): # d [(x-x_mean)/x_std] / dx
    BS = x.shape[0]
    N = x.shape[-1]
    
    jac = normalized_x_bkwd_rowwise(x)
    jac = torch.block_diag(*jac.unbind(0)).reshape(BS, N, BS, N)
    return jac

def normalized_x_bkwd_rowwise(x): # d [(x-x_mean)/x_std] / dx
    # f(x) = x - x_mean, g(x) = x_std
    # Note, below is "shorten Jacobian": rows are independent, so zeros in result are skipped.
    def std_bkwd(x):
        N = x.shape[-1]
        x_mean = torch.mean(x, axis=-1, keepdims=True)
        x_std = torch.std(x, axis=-1, keepdims = True)
        return 1 / (x_std * (N-1)) * (x - x_mean)

    BS, N = x.shape
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims = True)
     
    x_eye = torch.eye(N, device=x.device).expand(BS, N, N)
    jac = (x_eye - 1/N) *x_std.unsqueeze(-1) # fdx_g
    jac.sub_(torch.matmul((x-x_mean).unsqueeze(-1), std_bkwd(x).unsqueeze(-2))) # - f_gdx
    jac.mul_(1/torch.pow(x_std, 2).unsqueeze(-1)) # * g_pow2
    return jac

# Note that there is one semantic difference between this 
# and normalized_x_bkwd_rowwise (beside vjp):
# normalized_x_bkwd_rowwise returns jacobian which needs to be transposed.
def normalized_x_bkwd2(dloss_dx, x): # d [(x-x_mean)/x_std] / dx
    # f(x) = x - x_mean, g(x) = x_std
    BS, N = x.shape
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_rstd = 1/torch.std(x, axis=-1, keepdims = True)
    x_norm = (x - x_mean) * x_rstd
    
    x_eye = torch.eye(N, device=x.device).expand(BS, N, N)
    f_gdx =  torch.matmul(x_norm.unsqueeze(-1), x_norm.unsqueeze(-2)/(N-1)) 
    jac = (x_eye - 1/N - f_gdx) *x_rstd.unsqueeze(-1)
    
    return _vjp_in_2d_rowise(dloss_dx, jac.transpose(-2,-1)) 

# TODO XXX XXX: investigate why this is more memory efficient than my implementation above
# (Inspired from llm.c)
def normalized_x_bkwd2_plus(dloss_dx, x): # d [(x-x_mean)/x_std] / dx
    # f(x) = x - x_mean, g(x) = x_std
    BS, N = x.shape
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_rstd = 1/torch.std(x, axis=-1, keepdims = True)
    x_norm = (x - x_mean) * x_rstd
    
    n_adj = N/(N-1)
    dloss_dx = dloss_dx - dloss_dx.mean(-1, keepdim=True) - x_norm * (dloss_dx * x_norm).mean(-1, keepdim=True) * n_adj
    dloss_dx *= x_rstd
    return dloss_dx

def t_layernorm_bkwd_x(layer_params, x):
    x_2d = x.reshape((-1, x.shape[-1]))
    jac_x_2d = (layer_params[0] * normalized_x_bkwd(x_2d)).transpose(-3,-1)
    return jac_x_2d.reshape(x.shape + x.shape)

def t_layernorm_bkwd2_x(dloss_dx, layer_params, x):
    x_2d = x.reshape((-1, x.shape[-1]))
    # TODO XXX XXX: investigate the difference in memory consumption between two
    #return normalized_x_bkwd2(dloss_dx * layer_params[0], x_2d)
    dloss_dx_2d = dloss_dx.reshape((-1, dloss_dx.shape[-1]))
    return normalized_x_bkwd2_plus(dloss_dx_2d * layer_params[0], x_2d).reshape(dloss_dx.shape)
    
def t_gpt2_tlayer_sublock1_fwd(layer_params, y, mask, train=True, p_gen_aux=None):
    if not train:
        p_gen_aux = [None, None]
        
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y = y + t_dropout_fwd(t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0]), train, p_gen_aux[1])
    return y

def t_gpt2_tlayer_sublock1_fwd3(layer_params, y, mask, train=True, p_gen_aux=None):
    if not train:
        p_gen_aux = [None, None]
        
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff1 = y_diff
    y_diff, y_diff_acts = t_tlayer_attn_fwd3(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    acts = [(y_diff1, y_diff_acts)]
    acts.append(y_diff)    
    y = y + t_dropout_fwd(y_diff, train, p_gen_aux[1])
    return y, acts

def t_gpt2_tlayer_sublock1_bkwd_p(layer_params, y, mask, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    if not train:
        p_gen_aux = [None, None]
        
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_p = t_layernorm_bkwd_p(layer_params[:2], y)
    y_diff_attn = t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    y = y + t_dropout_fwd(y_diff_attn, train, p_gen_aux[1])

    jac_dropout = t_dropout_bkwd(y_diff_attn, train, p_gen_aux[1])
    jac_tlayer_attn_p = t_tlayer_attn_bkwd_p(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    jac_tlayer_attn_x = t_tlayer_attn_bkwd_x(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    
    jac_tlayer_attn_p = _mult_jacs_in_2d(jac_dropout, jac_tlayer_attn_p, y_diff_attn)
    jac_tlayer_attn_x = _mult_jacs_in_2d(jac_dropout, jac_tlayer_attn_x, y_diff_attn)
    
    jac_tlayer_attn_x = torch.stack(jac_tlayer_attn_x)
    jac_layernorm_p = [torch.einsum("xabcdef, defg->abcg", jac_tlayer_attn_x, j) for j in jac_layernorm_p]
    return tuple(jac_layernorm_p + jac_tlayer_attn_p)

def t_gpt2_tlayer_sublock1_bkwd2_p(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    if not train:
        p_gen_aux = [None, None]
        
    y_in = y
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_attn = t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    y = y + t_dropout_fwd(y_diff_attn, train, p_gen_aux[1])

    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_attn, train, p_gen_aux[1])
    tlayer_attn_dloss_dp = t_tlayer_attn_bkwd2_p(dloss_dx, layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])   
    dloss_dx = t_tlayer_attn_bkwd2_x(dloss_dx, layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    dloss_dx = torch.stack(dloss_dx).sum(dim=0) # TODO XXX: is there more efficient way of writing it?
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y_in)
    
    return layernorm_dloss_dp + tlayer_attn_dloss_dp

def t_gpt2_tlayer_sublock1_bkwd_x(layer_params, y, mask, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    if not train:
        p_gen_aux = [None, None]
    
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_x = t_layernorm_bkwd_x(layer_params[:2], y)
    y_diff_attn = t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    y = y + t_dropout_fwd(y_diff_attn, train, p_gen_aux[1])

    jac_dropout = t_dropout_bkwd(y_diff_attn, train, p_gen_aux[1])
    jac_tlayer_attn_x = t_tlayer_attn_bkwd_x(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    
    jac_tlayer_attn_x = _mult_jacs_in_2d(jac_dropout, jac_tlayer_attn_x, y_diff_attn)
    
    jac_y = torch.eye(y.numel(), device=y.device)    
    jac_tlayer_attn_x = torch.stack(jac_tlayer_attn_x)
    jac_y_diff = torch.einsum("xabcdef, defghi->abcghi", jac_tlayer_attn_x, jac_layernorm_x)
    return jac_y.reshape(jac_y_diff.shape) + jac_y_diff

def t_gpt2_tlayer_sublock1_bkwd2_x(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    if not train:
        p_gen_aux = [None, None]
        
    y_in=y
    blck_dloss_dx = dloss_dx
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_attn = t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    y = y + t_dropout_fwd(y_diff_attn, train, p_gen_aux[1])

    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_attn, train, p_gen_aux[1])
    dloss_dx = t_tlayer_attn_bkwd2_x(dloss_dx, layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    dloss_dx = torch.stack(dloss_dx).sum(dim=0)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y_in)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    jac_y = torch.eye(y.numel(), device=y.device).reshape(blck_dloss_dx.shape + blck_dloss_dx.shape)
    dloss_dx = _vjp_in_2d(blck_dloss_dx, jac_y) + dloss_dx
    
    return dloss_dx

def t_gpt2_tlayer_sublock1_bkwd2(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    if not train:
        p_gen_aux = [None, None]
        
    y_in=y
    blck_dloss_dx = dloss_dx
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_attn = t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    y = y + t_dropout_fwd(y_diff_attn, train, p_gen_aux[1])

    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_attn, train, p_gen_aux[1])
    dloss_dx, tlayer_attn_dloss_dp = t_tlayer_attn_bkwd2(dloss_dx, layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    dloss_dx = torch.stack(dloss_dx).sum(dim=0)
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y_in)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y_in)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    dloss_dx = blck_dloss_dx + dloss_dx
    dloss_dp = layernorm_dloss_dp + tlayer_attn_dloss_dp
    
    return dloss_dx, dloss_dp

def t_gpt2_tlayer_sublock1_bkwd3(dloss_dx, acts, layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None]
        
    blck_dloss_dx = dloss_dx
    y_diff = acts[0][0]
    y_diff_attn = acts[1]

    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_attn, train, p_gen_aux[1])
    dloss_dx, tlayer_attn_dloss_dp = t_tlayer_attn_bkwd3(dloss_dx, acts[0][1], layer_params[2:], (y_diff, y_diff, y_diff), mask, train, p_gen_aux[0])
    dloss_dx = torch.stack(dloss_dx).sum(dim=0)
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    dloss_dx = blck_dloss_dx + dloss_dx
    dloss_dp = layernorm_dloss_dp + tlayer_attn_dloss_dp
    
    return dloss_dx, dloss_dp
    
def t_gpt2_tlayer_sublock2_fwd(layer_params, y, train=True, p_gen_aux=None):
    y_diff = t_layernorm_fwd(layer_params[:-4], y)
    y = y + t_dropout_fwd(t_tlayer_ffn_fwd(layer_params[-4:], y_diff, t_gelu_fwd), train, p_gen_aux)
    return y

def t_gpt2_tlayer_sublock2_fwd3(layer_params, y, train=True, p_gen_aux=None):
    y_diff = t_layernorm_fwd(layer_params[:-4], y)
    acts = [y_diff]
    y_diff = t_tlayer_ffn_fwd(layer_params[-4:], y_diff, t_gelu_fwd)
    acts.append(y_diff)
    # TODO XXX XXX: The below line (i.e. with activation checkpointing) is not faster (due to reshapes?)
    # Remember to stick to acts convention: each element of acts should represent single op's acts and input,
    # so we should create a tuple of ("y_diff from above", ffn_acts) as first element of acts.
    #y_diff, ffn_acts = t_tlayer_ffn_fwd3(layer_params[-4:], y_diff, t_gelu_fwd)
    #acts.append((y_diff, ffn_acts))
    y = y + t_dropout_fwd(y_diff, train, p_gen_aux)
    return y, acts

def t_gpt2_tlayer_sublock2_bkwd_p(layer_params, y, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_p = t_layernorm_bkwd_p(layer_params[:2], y)
    y_diff_ffn = t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd)
    y = y + t_dropout_fwd(y_diff_ffn, train, p_gen_aux)
    
    jac_dropout = t_dropout_bkwd(y_diff_ffn, train, p_gen_aux)
    jac_tlayer_ffn_p = t_tlayer_ffn_bkwd_p(layer_params[2:], y_diff, t_gelu_fwd)
    jac_tlayer_ffn_x = t_tlayer_ffn_bkwd_x(layer_params[2:], y_diff, t_gelu_fwd)
      
    jac_tlayer_ffn_p = _mult_jacs_in_2d(jac_dropout, jac_tlayer_ffn_p, y_diff_ffn)
    jac_tlayer_ffn_x = _mult_jacs_in_2d(jac_dropout, [jac_tlayer_ffn_x], y_diff_ffn)[0]
    
    jac_layernorm_p = [torch.einsum("abcdef, defg->abcg", jac_tlayer_ffn_x, j) for j in jac_layernorm_p]
    return tuple(jac_layernorm_p + jac_tlayer_ffn_p)

def t_gpt2_tlayer_sublock2_bkwd2_p(dloss_dx, layer_params, y, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    y_in = y
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_ffn = t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd)
    y = y + t_dropout_fwd(y_diff_ffn, train, p_gen_aux)
    
    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_ffn, train, p_gen_aux)
    tlayer_ffn_dloss_dp = t_tlayer_ffn_bkwd2_p(dloss_dx, layer_params[2:], y_diff, t_gelu_fwd)
    dloss_dx = t_tlayer_ffn_bkwd2_x(dloss_dx, layer_params[2:], y_diff, t_gelu_fwd)
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y_in)
    
    return layernorm_dloss_dp + tlayer_ffn_dloss_dp

def t_gpt2_tlayer_sublock2_bkwd_x(layer_params, y, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_x = t_layernorm_bkwd_x(layer_params[:2], y)
    y_diff_ffn = t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd)
    y = y + t_dropout_fwd(y_diff_ffn, train, p_gen_aux)
    
    jac_dropout = t_dropout_bkwd(y_diff_ffn, train, p_gen_aux)
    jac_tlayer_ffn_x = t_tlayer_ffn_bkwd_x(layer_params[2:], y_diff, t_gelu_fwd)
    
    # TODO XXX: Figure out how to reliably test addition of the below line
    jac_tlayer_ffn_x = _mult_jacs_in_2d(jac_dropout, [jac_tlayer_ffn_x], y_diff_ffn)[0]
    
    jac_y = torch.eye(y.numel(), device=y.device)    
    jac_y_diff = torch.einsum("abcdef, defghi->abcghi", jac_tlayer_ffn_x, jac_layernorm_x)
    return jac_y.reshape(jac_y_diff.shape) + jac_y_diff

def t_gpt2_tlayer_sublock2_bkwd2_x(dloss_dx, layer_params, y, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    y_in = y
    blck_dloss_dx = dloss_dx
    
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_ffn = t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd)
    y = y + t_dropout_fwd(y_diff_ffn, train, p_gen_aux)
    
    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_ffn, train, p_gen_aux)
    dloss_dx = t_tlayer_ffn_bkwd2_x(dloss_dx, layer_params[2:], y_diff, t_gelu_fwd)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y_in)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    jac_y = torch.eye(y.numel(), device=y.device).reshape(blck_dloss_dx.shape +blck_dloss_dx.shape)    
    dloss_dx = _vjp_in_2d(blck_dloss_dx, jac_y) + dloss_dx
    
    return dloss_dx

def t_gpt2_tlayer_sublock2_bkwd2(dloss_dx, layer_params, y, train=True, p_gen_aux=None): # input: seq_len x emb_dim
    y_in = y
    blck_dloss_dx = dloss_dx
    
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y_diff_ffn = t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd)
    y = y + t_dropout_fwd(y_diff_ffn, train, p_gen_aux)
    
    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_ffn, train, p_gen_aux)
    dloss_dx, tlayer_ffn_dloss_dp = t_tlayer_ffn_bkwd2(dloss_dx, layer_params[2:], y_diff, t_gelu_fwd)
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y_in)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y_in)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    dloss_dx = blck_dloss_dx + dloss_dx
    dloss_dp = layernorm_dloss_dp + tlayer_ffn_dloss_dp
    
    return dloss_dx, dloss_dp

def t_gpt2_tlayer_sublock2_bkwd3(dloss_dx, acts, layer_params, y, train=True, p_gen_aux=None): # input: N x D
    blck_dloss_dx = dloss_dx
    
    y_diff = acts[0]
    y_diff_ffn = acts[1]
    #y_diff_ffn = acts[1][0] (with activation checkpointing)
    
    # propagate back
    dloss_dx = t_dropout_bkwd2(dloss_dx, y_diff_ffn, train, p_gen_aux)
    dloss_dx, tlayer_ffn_dloss_dp = t_tlayer_ffn_bkwd2(dloss_dx, layer_params[2:], y_diff, t_gelu_fwd)
    # TODO XXX XXX: The below line (i.e. with activation checkpointing) is not faster (due to reshapes?)
    #dloss_dx, tlayer_ffn_dloss_dp = t_tlayer_ffn_bkwd3(dloss_dx, acts[1][1], layer_params[2:], y_diff, t_gelu_fwd)
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, layer_params[:2], y)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, layer_params[:2], y)
    # account for "y" in residual's "y + y_diff". TODO XXX: Does this reshape make sense?
    dloss_dx = blck_dloss_dx + dloss_dx
    dloss_dp = layernorm_dloss_dp + tlayer_ffn_dloss_dp
    
    return dloss_dx, dloss_dp

def t_gpt2_tlayer_fwd(layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None] 

    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    y = t_gpt2_tlayer_sublock2_fwd(layer_params[-6:], y, train, p_gen_aux[2])
    return y

def t_gpt2_tlayer_fwd3(layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None] 

    y, sblck1_acts = t_gpt2_tlayer_sublock1_fwd3(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    acts = [sblck1_acts]
    sblck2_y = y
    y, sblck2_acts = t_gpt2_tlayer_sublock2_fwd3(layer_params[-6:], y, train, p_gen_aux[2])
    acts.append((sblck2_y, sblck2_acts))
    return y, acts

def t_gpt2_tlayer_bkwd_p(layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None] 
        
    jac_subblock1_p = t_gpt2_tlayer_sublock1_bkwd_p(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    jac_subblock2_p = t_gpt2_tlayer_sublock2_bkwd_p(layer_params[-6:], y, train, p_gen_aux[2])
    jac_subblock2_x = t_gpt2_tlayer_sublock2_bkwd_x(layer_params[-6:], y, train, p_gen_aux[2])
    
    jac_subblock1_p = _mult_jacs_in_2d(jac_subblock2_x, jac_subblock1_p, y)
    return tuple(jac_subblock1_p) + jac_subblock2_p

def t_gpt2_tlayer_bkwd2_p(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None] 
        
    y_in = y
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    subblock2_dloss_dp = t_gpt2_tlayer_sublock2_bkwd2_p(dloss_dx, layer_params[-6:], y, train, p_gen_aux[2])
    dloss_dx = t_gpt2_tlayer_sublock2_bkwd2_x(dloss_dx, layer_params[-6:], y, train, p_gen_aux[2])
    subblock1_dloss_dp = t_gpt2_tlayer_sublock1_bkwd2_p(dloss_dx, layer_params[:-6], y_in, mask, train, p_gen_aux[:2])
    
    return subblock1_dloss_dp + subblock2_dloss_dp

def t_gpt2_tlayer_bkwd_x(layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None]    
    
    jac_subblock1_x = t_gpt2_tlayer_sublock1_bkwd_x(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    jac_subblock2_x = t_gpt2_tlayer_sublock2_bkwd_x(layer_params[-6:], y, train, p_gen_aux[2])
      
    return torch.einsum('abcdef, defghi->abcghi', jac_subblock2_x, jac_subblock1_x)

def t_gpt2_tlayer_bkwd2_x(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None]    
    
    y_in = y
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    dloss_dx = t_gpt2_tlayer_sublock2_bkwd2_x(dloss_dx, layer_params[-6:], y, train, p_gen_aux[2])
    
    return t_gpt2_tlayer_sublock1_bkwd2_x(dloss_dx, layer_params[:-6], y_in, mask, train, p_gen_aux[:2])  

def t_gpt2_tlayer_bkwd2(dloss_dx, layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None]    
    
    y_in = y
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train, p_gen_aux[:2])
    dloss_dx, subblock2_dloss_dp = t_gpt2_tlayer_sublock2_bkwd2(dloss_dx, layer_params[-6:], y, train, p_gen_aux[2])
    dloss_dx, subblock1_dloss_dp = t_gpt2_tlayer_sublock1_bkwd2(dloss_dx, layer_params[:-6], y_in, mask, train, p_gen_aux[:2])
    dloss_dp = subblock1_dloss_dp + subblock2_dloss_dp
    
    return dloss_dx, dloss_dp

def t_gpt2_tlayer_bkwd3(dloss_dx, acts, layer_params, y, mask, train=True, p_gen_aux=None): # input: N x D
    if not train:
        p_gen_aux = [None, None, None]    
    
    dloss_dx, subblock2_dloss_dp = t_gpt2_tlayer_sublock2_bkwd3(dloss_dx, acts[-1][1], layer_params[-6:], acts[-1][0], train, p_gen_aux[2])
    dloss_dx, subblock1_dloss_dp = t_gpt2_tlayer_sublock1_bkwd3(dloss_dx, acts[-2], layer_params[:-6], y, mask, train, p_gen_aux[:2])
    dloss_dp = subblock1_dloss_dp + subblock2_dloss_dp
    
    return dloss_dx, dloss_dp

def t_gpt2_tlayers_fwd(params, y, mask, indices, train=True, p_gen_aux=None): # input: seq_len x
    if not train: # as there are 3 dropouts per tlayer
        p_gen_aux = [None] + [None] * 3 * (len(params) - 3)
    
    y = t_embed_fwd(params[0], y)
    y = t_dropout_fwd(y + params[1][0], train, p_gen_aux[0])
    
    for i, layer_params in enumerate(params[2:-1]):
        layer_p_gen_aux = p_gen_aux[1+i*3:1+(i+1)*3]
        y = t_gpt2_tlayer_fwd(layer_params, y, mask, train, layer_p_gen_aux)
    y = t_layernorm_fwd(params[-1], y)

    return y

def t_gpt2_tlayers_fwd3(params, y, mask, indices, train=True, p_gen_aux=None): # input: seq_len x
    if not train: # as there are 3 dropouts per tlayer
        p_gen_aux = [None] + [None] * 3 * (len(params) - 3)
    
    y = t_embed_fwd(params[0], y)
    y = y + params[1][0]
    acts = [y]
    y = t_dropout_fwd(y, train, p_gen_aux[0])
    
    for i, layer_params in enumerate(params[2:-1]):
        layer_p_gen_aux = p_gen_aux[1+i*3:1+(i+1)*3]
        layer_input = y
        y, layer_acts = t_gpt2_tlayer_fwd3(layer_params, y, mask, train, layer_p_gen_aux)
        acts.append([layer_input, layer_acts])
    acts.append(y)
    y = t_layernorm_fwd(params[-1], y)

    return y, acts

# Multiplies (in 2D) left Jacobian against the nested list of right Jacobians
# Uses y for doing reshapes to 2D correctly, but probably one doesn't need it
# TODO XXX: func should support PyTree at right
def _mult_jacs_in_2d(j_left, j_right_tree, y_in):
    # As j_left.shape = y_out.shape + y_in.shape
    y_out_shape = j_left.flatten(start_dim=-len(y_in.shape)).shape[:-1]
    
    def mult_j_in_2d(j_left_2d, j): # we need to do it u
        # As j.shape = y_in.shape + j_in.shape
        j_in_shape = j.flatten(end_dim=len(y_in.shape)-1).shape[1:]
        j_2d = j.reshape((y_in.numel(), -1))
        return torch.matmul(j_left_2d, j_2d).reshape(y_out_shape + j_in_shape)
    j_left_2d = j_left.reshape((-1, y_in.numel()))
    return [mult_j_in_2d(j_left_2d, j) for j in j_right_tree] 

def t_gpt2_tlayers_bkwd_p(params, y, mask, indices, train=True, p_gen_aux=None): # input: seq_len x
    if not train: # as there are 3 dropouts per tlayer
        p_gen_aux = [None] + [None] * 3 * (len(params) - 3)    
    
    indices = torch.arange(y.shape[1], device=y.device).unsqueeze(0).expand(*y.shape) # we ignore indices arg
    jac_embed = t_embed_bkwd(params[0], y)
    # Due to tying of embedding and final projection layers,
    # we need to fill zeroed gradient with respect to biases:
    jac_embed = [jac_embed[0], torch.zeros(jac_embed[0].shape[:-1], device=y.device)]
    y = t_embed_fwd(params[0], y)
    # Reuse t_embed_bkwd to compute jacobian of pos_encoding
    # Need to account for lack of  1/ sqrt(emb_dim)
    jac_pos_enc = list(t_embed_bkwd(params[1], indices))
    jac_pos_enc[0][jac_pos_enc[0]!=0] = 1
    jac_dropout = t_dropout_bkwd(y + params[1][0], train, p_gen_aux[0])
    y = t_dropout_fwd(y + params[1][0], train, p_gen_aux[0])
    
    layers_jacs_p = []
    layers_jacs_x = []
    
    for i, layer_params in enumerate(params[2:-1]):
        layer_p_gen_aux = p_gen_aux[1+i*3:1+(i+1)*3]
        layers_jacs_p.append(t_gpt2_tlayer_bkwd_p(layer_params, y, mask, train, layer_p_gen_aux))
        layers_jacs_x.append(t_gpt2_tlayer_bkwd_x(layer_params, y, mask, train, layer_p_gen_aux))
        y = t_gpt2_tlayer_fwd(layer_params, y, mask, train, layer_p_gen_aux)
    jac_layernorm_p = t_layernorm_bkwd_p(params[-1], y)
    jac_layernorm_x = t_layernorm_bkwd_x(params[-1], y)    
    y = t_layernorm_fwd(params[-1], y)
    
    # Propoagate back
    layers_jacs_x[-1]=torch.einsum('abcdef, defghi -> abcghi', jac_layernorm_x, layers_jacs_x[-1])
    layers_jacs_p[-1] = _mult_jacs_in_2d(jac_layernorm_x, layers_jacs_p[-1], y)
    for i in reversed(range(1, len(layers_jacs_p))):
        layers_jacs_x[i-1]=torch.einsum('abcdef, defghi -> abcghi',layers_jacs_x[i], layers_jacs_x[i-1])
        layers_jacs_p[i-1] = _mult_jacs_in_2d(layers_jacs_x[i], layers_jacs_p[i-1], y)
    jac_dropout = torch.einsum('abcdef, defghi -> abcghi', layers_jacs_x[0], jac_dropout)
    jac_pos_enc[0] =torch.einsum('abcdef, defgh -> abcgh', jac_dropout, jac_pos_enc[0])
    jac_embed[0] = torch.einsum('abcdef, defgh -> abcgh', jac_dropout, jac_embed[0])
    # Note, no need to propagate for jac_embed[1], since it's zeroeed 

    return tuple([jac_embed, jac_pos_enc] + layers_jacs_p + [jac_layernorm_p])

def t_gpt2_tlayers_bkwd2_p(dloss_dx, params, y, mask, indices, train=True, p_gen_aux=None): # input: seq_len x
    if not train: # as there are 3 dropouts per tlayer
        p_gen_aux = [None] + [None] * 3 * (len(params) - 3)    
    
    y_in = y
    indices = torch.arange(y.shape[1], device=y.device).unsqueeze(0).expand(*y.shape) # we ignore indices arg
    y = t_embed_fwd(params[0], y)
    t_dropout_input = y + params[1][0]
    y = t_dropout_fwd(y + params[1][0], train, p_gen_aux[0])
    
    layers_inputs = []
    for i, layer_params in enumerate(params[2:-1]):
        layer_p_gen_aux = p_gen_aux[1+i*3:1+(i+1)*3]
        layers_inputs.append((y, layer_p_gen_aux))
        y = t_gpt2_tlayer_fwd(layer_params, y, mask, train, layer_p_gen_aux)
    
    # Propoagate back    
    # layernorm
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, params[-1], y)
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, params[-1], y) 
    
    # layers
    layers_dloss_dp = []
    for i, layer_params in reversed(list(enumerate(params[2:-1]))):
        y, layer_p_gen_aux = layers_inputs[i]
        # Use bkwd2 which combines dloss_dx and dloss_dp computations (for efficiency reasons)
        # TODO XXX: do sanity check whether the results are exactly the same as for separate
        # bkwd2_p and bkwd2_x
        dloss_dx, layer_dloss_dp = t_gpt2_tlayer_bkwd2(dloss_dx, layer_params, y, mask, train, layer_p_gen_aux)
        layers_dloss_dp.append(layer_dloss_dp)
        #layers_dloss_dp.append(t_gpt2_tlayer_bkwd2_p(dloss_dx, layer_params, y, mask, train, layer_p_gen_aux))
        #dloss_dx = t_gpt2_tlayer_bkwd2_x(dloss_dx, layer_params, y, mask, train, layer_p_gen_aux)
    layers_dloss_dp = list(reversed(layers_dloss_dp)) # TODO XXX: clean up list+ reversed combos
    
    # dropout + embed + pos_enc
    dloss_dx = t_dropout_bkwd2(dloss_dx, t_dropout_input, train, p_gen_aux[0])
    embed_dloss_dp = t_embed_bkwd2(dloss_dx, params[0], y_in)
    # Due to tying of embedding and final projection layers,
    # we need to fill zeroed gradient with respect to biases:
    embed_dloss_dp = [embed_dloss_dp[0], torch.zeros(embed_dloss_dp[0].shape[:-1], device=y_in.device)]
    pos_enc_dloss_dp = t_indexing_bkwd2(dloss_dx, params[1], indices)

    dloss_dp = [embed_dloss_dp, pos_enc_dloss_dp] + layers_dloss_dp + [layernorm_dloss_dp]
    return tuple(dloss_dp)

def t_gpt2_tlayers_bkwd3_p(dloss_dx, acts, params, y, mask, indices, train=True, p_gen_aux=None): # input: seq_len x
    if not train: # as there are 3 dropouts per tlayer
        p_gen_aux = [None] + [None] * 3 * (len(params) - 3)    
    
    indices = torch.arange(y.shape[1], device=y.device).unsqueeze(0).expand(*y.shape) # we ignore indices arg
    t_dropout_input = acts[0]
    layers_p_gen_aux = [p_gen_aux[1+i*3:1+(i+1)*3] for i in range(len(params) - 3)]
    layers_inputs = list(zip(acts[1:-1], layers_p_gen_aux))
    
    # Propoagate back    
    # layernorm
    layernorm_dloss_dp = t_layernorm_bkwd2_p(dloss_dx, params[-1], acts[-1])
    dloss_dx = t_layernorm_bkwd2_x(dloss_dx, params[-1], acts[-1]) 
    
    # layers
    layers_dloss_dp = []
    for i, layer_params in reversed(list(enumerate(params[2:-1]))):
        layer_acts, layer_p_gen_aux = layers_inputs[i]
        # TODO XXX: Note in bkwd2, there was a comment to compare bkwd against bkwd2_x, bkwd2_p combined. I think it can be ignored
        dloss_dx, layer_dloss_dp = t_gpt2_tlayer_bkwd3(dloss_dx, layer_acts[1], layer_params, layer_acts[0], mask, train, layer_p_gen_aux)
        layers_dloss_dp.append(layer_dloss_dp)
    layers_dloss_dp = list(reversed(layers_dloss_dp)) # TODO XXX: clean up list+ reversed combos
    
    # dropout + embed + pos_enc
    dloss_dx = t_dropout_bkwd2(dloss_dx, t_dropout_input, train, p_gen_aux[0])
    embed_dloss_dp = t_embed_bkwd2(dloss_dx, params[0], y)
    # Due to tying of embedding and final projection layers,
    # we need to fill zeroed gradient with respect to biases:
    embed_dloss_dp = [embed_dloss_dp[0], torch.zeros(embed_dloss_dp[0].shape[:-1], device=y.device)]
    pos_enc_dloss_dp = t_indexing_bkwd2(dloss_dx, params[1], indices)

    dloss_dp = [embed_dloss_dp, pos_enc_dloss_dp] + layers_dloss_dp + [layernorm_dloss_dp]
    return tuple(dloss_dp)

def t_gpt2_forward(params, y, y_mask, y_indices, train, p_gen_aux=None): # input: seq_len x
    y = t_gpt2_tlayers_fwd(params, y, y_mask, y_indices, train, p_gen_aux)
    
    y = t_linear_fwd(params[0], y) 
    return y

def t_gpt2_forward_with_acts(params, y, y_mask, y_indices, train, p_gen_aux=None): # input: seq_len x
    y, acts = t_gpt2_tlayers_fwd3(params, y, y_mask, y_indices, train, p_gen_aux)
    y1 = y
    y = t_linear_fwd(params[0], y) 
    return y, [acts, y1]

def t_gpt2_bkwd_p(params, y, y_mask, y_indices, train, p_gen_aux=None): # input: seq_len x
    jac = t_gpt2_tlayers_bkwd_p(params, y, y_mask, y_indices, train, p_gen_aux)
    y = t_gpt2_tlayers_fwd(params, y, y_mask, y_indices, train, p_gen_aux)
    
    jac_linear_x = t_linear_bkwd_x(params[0], y) 
    jac_linear_p = t_linear_bkwd_p(params[0], y)    
    
    jac = list(jac)
    for i in range(len(jac)):
        jac[i] = _mult_jacs_in_2d(jac_linear_x, jac[i], y)
    
    # As we tie embedding and last projection weights (no need to add jac[0][1] as it's zeroed)
    jac[0] = (jac[0][0] + jac_linear_p[0], jac_linear_p[1])
    return tuple(jac)

def t_gpt2_bkwd2_p(dloss_dx, params, y, y_mask, y_indices, train, p_gen_aux=None): # input: seq_len x
    y0 = y
    y = t_gpt2_tlayers_fwd(params, y, y_mask, y_indices, train, p_gen_aux)
    
    linear_dloss_dp = t_linear_bkwd2_p(dloss_dx, params[0], y)    
    dloss_dx = t_linear_bkwd2_x(dloss_dx, params[0], y)
    
    dloss_dp = t_gpt2_tlayers_bkwd2_p(dloss_dx, params, y0, y_mask, y_indices, train, p_gen_aux)    
    dloss_dp = list(dloss_dp)
    
    # As we tie embedding and last projection weights (no need to add jac[0][1] as it's zeroed)
    dloss_dp[0] = (dloss_dp[0][0] + linear_dloss_dp[0], linear_dloss_dp[1])
        
    return tuple(dloss_dp)

def t_gpt2_bkwd3_p(dloss_dx, acts, params, y, y_mask, y_indices, train, p_gen_aux=None): # input: seq_len x
    linear_dloss_dp = t_linear_bkwd2_p(dloss_dx, params[0], acts[-1])    
    dloss_dx = t_linear_bkwd2_x(dloss_dx, params[0], acts[-1])
    
    dloss_dp = t_gpt2_tlayers_bkwd3_p(dloss_dx, acts[0], params, y, y_mask, y_indices, train, p_gen_aux)    
    dloss_dp = list(dloss_dp)
    
    # As we tie embedding and last projection weights (no need to add jac[0][1] as it's zeroed)
    dloss_dp[0] = (dloss_dp[0][0] + linear_dloss_dp[0], linear_dloss_dp[1])
        
    return tuple(dloss_dp)

t_batched_forward_gpt2 = t_gpt2_forward # TODO XXX: replace the references to the left with the right
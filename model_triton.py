# WORK IN PROGRESS: writing derivatives right now

### PARAMS + MODEL
DROPOUT_RATE = 0.1 # TODO: move it out, and pass as paramteter

import math
import torch

### PARAMS: they are the same as for Torch.Func, so import

from model_torch_func import init_transformer_gpt2, count_num_params, batched_forward_gpt2

### MODEL in TRITON

def t_log_softmax_fwd(x_logits): # compute log_softmax from logits over the last dimension
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0] # as it returns (maxs, indices)
    return x_logits - torch.logsumexp(x_logits, axis=-1, keepdims=True)

# Other module assumes different name
# TODO XXX: fix references in other file
log_softmax = t_log_softmax_fwd 

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

def t_linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params[0], 0, 1)) + layer_params[1][None, :] # since layer_params[0] is output_dim x emb_dim, layer_params[1] is output_dim

def t_linear_bkwd_p(layer_params, x): # input: seq_len x emb_dim
    x_indim = x.shape[0] # TODO: support x > 2D
    outdim = layer_params[1].shape[0]

    jac1 = t_proj_bkwd_p(layer_params[0], x)
    jac2 = torch.eye(outdim, device=x.device).expand((x_indim, outdim, outdim))
    return jac1, jac2

def t_linear_bkwd_x(layer_params, x): # input: seq_len x emb_dim
    return t_proj_bkwd_x(layer_params[0], x)

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

# TODO XXX: WIP
def t_softmax_attn_fwd(q, k, mask, train):
    D = q.shape[-1]
    attn = torch.matmul(q, torch.transpose(k, -2, -1))
    attn = attn / math.sqrt(D)
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = torch.exp(t_log_softmax_fwd(attn))
    softmaxed_attn = t_dropout(softmaxed_attn, train)
    return softmaxed_attn

# TODO XXX: WIP 
def t_softmax_attn_bkwd(q, k, mask, train):
    D = q.shape[-1]
    attn = torch.matmul(q, torch.transpose(k, -2, -1))
    attn = attn / math.sqrt(D)
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = torch.exp(t_log_softmax_fwd(attn)) # TODO XXX: numerical stability
    softmaxed_attn = t_dropout(softmaxed_attn, train)
    
    d_dropout_dx = 1 #0.9 # TODO: XXX add proper dropout 
    dsoftmaxed_attn_dx = d_dropout_dx * softmaxed_attn[..., None, None, None, None] * t_log_softmax_bkwd(attn)
    jac1 = torch.matmul(dsoftmaxed_attn_dx, k/math.sqrt(D))
    jac2 = torch.matmul(q.transpose(-2,-1)/math.sqrt(D), dsoftmaxed_attn_dx).transpose(-2,-1)
    return jac1, jac2

# TODO XXX: Use t_softmax_attn_fwd above?
def t_scaled_dot_prod_attn_fwd(qkv, mask, train=True): # inputs: batch_size x heads x 3 x seq_len x emb_dim, mask: batch_size x seq_len(q) x seq_len(k)
    q, k, v = torch.unbind(qkv, dim=2)# batch_size x heads x seq_len x emb_dim
    attn = torch.matmul(q, torch.transpose(k, -2, -1)) # seq_len(q) x seq_len(k)
    attn = attn / math.sqrt(q.shape[-1]) # scale by sqrt(d_k)
    #attn = jnp.where(mask, attn, jnp.full_like(attn, -jnp.inf))
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = torch.exp(t_log_softmax_fwd(attn))
    softmaxed_attn = t_dropout(softmaxed_attn, train)
    return torch.matmul(softmaxed_attn, v) # output: seq_len x emb_dim

# TODO XXX: Support for heads>1
# TODO XXX: replace mult with the generic newer _mult
def t_scaled_dot_prod_attn_bkwd(qkv, mask, train=True): # inputs: batch_size x heads x 3 x seq_len x emb_dim, mask: batch_size x seq_len(q) x seq_len(k)
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
        jac_ = jac_a.reshape(jac_a.shape[0], -1, v_2d.numel()).transpose(1, 0)
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

def t_tlayer_attn_heads_fwd(layer_params, qkv, mask, train): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    qkv = torch.stack(qkv,dim=-3) # batch_size x 3 x seq_len x emb_dim
    
    proj_qkv = t_proj_fwd(layer_params, torch.unsqueeze(qkv, 1)) # batch_size x heads x 3 x seq_len x emb_dim
    return t_scaled_dot_prod_attn_fwd(proj_qkv, mask, train)

# TODO XXX: This is placeholder
# 1. Code up Jacobian for bmm (which we effective use by calling t_proj_fwd)
# 2. There are some bugs in t_scaled_dot_prod_attn_bkwd when using diff/more dimensions
def t_tlayer_attn_heads_bkwd_p(layer_params, qkv, mask, train): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    from torch.func import jacrev
    from functools import partial 
    fn = partial(t_tlayer_attn_heads_fwd, mask=mask, train=False)
    return jacrev(fn)(layer_params, qkv)

# TODO XXX: same TODO as above
def t_tlayer_attn_heads_bkwd_x(layer_params, qkv, mask, train): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    from torch.func import jacrev
    from functools import partial 
    fn = partial(t_tlayer_attn_heads_fwd, mask=mask, train=False)
    return jacrev(fn, argnums=1)(layer_params, qkv)

def t_tlayer_attn_fwd(layer_params, qkv, mask, train): # input: batch_size x seq_len x emb_dim
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train)
    BS, H, N, D = heads_attns.shape
    attn = heads_attns.transpose(-2, -3).reshape((BS, N, -1))
    return t_proj_fwd(layer_params[-1], attn)

# TODO XXX: WIP make it work for H>1
def t_tlayer_attn_bkwd_p(layer_params, qkv, mask, train): # input: batch_size x seq_len x emb_dim
    f_out = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train)
    BS, H, N, D = f_out.shape
    g_in = f_out.transpose(-2, -3).reshape((BS, N, -1))
    jac_g_x = t_proj_bkwd_x(layer_params[-1], g_in)
    jac_g_p = t_proj_bkwd_p(layer_params[-1], g_in)
    
    f_out_size = f_out.numel()
    jac_f = t_tlayer_attn_heads_bkwd_p(layer_params[0], qkv, mask, train)
    jac_p0 = torch.matmul(jac_g_x.reshape((-1, f_out_size)), jac_f.reshape(f_out_size, -1))
    return jac_p0.reshape(g_in.shape + layer_params[0].shape), jac_g_p

# TODO XXX: WIP make it work for H>1
def t_tlayer_attn_bkwd_x(layer_params, qkv, mask, train): # input: batch_size x seq_len x emb_dim
    f_out = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train)
    BS, H, N, D = f_out.shape
    g_in = f_out.transpose(-2, -3).reshape((BS, N, -1))
    jac_g_x = t_proj_bkwd_x(layer_params[-1], g_in)
    
    f_out_size = f_out.numel()
    jac_f_x = t_tlayer_attn_heads_bkwd_x(layer_params[0], qkv, mask, train)
    
    # TODO XXX: vectorize?
    js = [torch.matmul(jac_g_x.reshape((-1, f_out_size)), j.reshape(f_out_size, -1)) for j in jac_f_x]
    return tuple([j.reshape(g_in.shape + qkv[0].shape) for j in js])

def t_tlayer_ffn_fwd(layer_params, x, activation_fn): # input: seq_len x emb_dim
    x = t_linear_fwd((layer_params[0], layer_params[1]), x)
    x = activation_fn(x)
    x = t_linear_fwd((layer_params[2], layer_params[3]), x)
    return x

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

def t_dropout(x, train=True):
    if not train: # As we jit the whole loss/inference, the train param is known at tracing time.
        return x * (1-DROPOUT_RATE)
    
    return x * torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE))

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

def normalized_x_bkwd(x): # d [(x-x_mean)/x_std] / dx
    # Note, below is "shorten Jacobian": rows are independent, so zeros in result are skipped.
    def std_bkwd(x): 
        N = x.shape[-1]
        x_mean = torch.mean(x, axis=-1, keepdims=True)
        x_std = torch.std(x, axis=-1, keepdims = True)
        return 1 / (x_std * (N-1)) * (x - x_mean)

    BS = x.shape[0]
    N = x.shape[-1]
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims = True)
     
    x_eye = torch.eye(N, device=x.device).expand(x.shape[0], N, N)
    fdx_g = (x_eye - 1/N) *x_std.unsqueeze(-1)
    f_gdx = torch.matmul((x-x_mean).unsqueeze(-1), std_bkwd(x).unsqueeze(-2)) 
    g_pow2 = 1/torch.pow(x_std, 2)

    jac = g_pow2.unsqueeze(-1) * (fdx_g  - f_gdx)
    return torch.block_diag(*jac.unbind(0)).reshape(BS, N, BS, N)

def t_layernorm_bkwd_x(layer_params, x):
    x_2d = x.reshape((-1, x.shape[-1]))
    jac_x_2d = (layer_params[0] * normalized_x_bkwd(x_2d)).transpose(-3,-1)
    return jac_x_2d.reshape(x.shape + x.shape)

def t_gpt2_tlayer_sublock1_fwd(layer_params, y, mask, train=True):
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train), train)
    return y

def t_gpt2_tlayer_sublock1_bkwd_p(layer_params, y, mask, train=True): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_p = t_layernorm_bkwd_p(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train), train)
    # TODO XXX: add dropout!
    jac_tlayer_attn_p = t_tlayer_attn_bkwd_p(layer_params[2:], (y_diff, y_diff, y_diff), mask, train)
    jac_tlayer_attn_x = t_tlayer_attn_bkwd_x(layer_params[2:], (y_diff, y_diff, y_diff), mask, train)
    
    jac_tlayer_attn_x = torch.stack(jac_tlayer_attn_x)
    jac_layernorm_p = [torch.einsum("xabcdef, defg->abcg", jac_tlayer_attn_x, j) for j in jac_layernorm_p]
    return tuple(jac_layernorm_p) + jac_tlayer_attn_p

def t_gpt2_tlayer_sublock1_bkwd_x(layer_params, y, mask, train=True): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_x = t_layernorm_bkwd_x(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_attn_fwd(layer_params[2:], (y_diff, y_diff, y_diff), mask, train), train)
    # TODO XXX: add dropout!
    jac_tlayer_attn_x = t_tlayer_attn_bkwd_x(layer_params[2:], (y_diff, y_diff, y_diff), mask, train)
    
    jac_y = torch.eye(y.numel(), device=y.device)    
    jac_tlayer_attn_x = torch.stack(jac_tlayer_attn_x)
    jac_y_diff = torch.einsum("xabcdef, defghi->abcghi", jac_tlayer_attn_x, jac_layernorm_x)
    return jac_y.reshape(jac_y_diff.shape) + jac_y_diff
    
def t_gpt2_tlayer_sublock2_fwd(layer_params, y, train=True):
    y_diff = t_layernorm_fwd(layer_params[:-4], y)
    y = y + t_dropout(t_tlayer_ffn_fwd(layer_params[-4:], y_diff, t_gelu_fwd), train)
    return y

def t_gpt2_tlayer_sublock2_bkwd_p(layer_params, y, train=True): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_p = t_layernorm_bkwd_p(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd), train)
    # TODO XXX: add dropout!
    jac_tlayer_ffn_p = t_tlayer_ffn_bkwd_p(layer_params[2:], y_diff, t_gelu_fwd)
    jac_tlayer_ffn_x = t_tlayer_ffn_bkwd_x(layer_params[2:], y_diff, t_gelu_fwd)
    
    jac_layernorm_p = [torch.einsum("abcdef, defg->abcg", jac_tlayer_ffn_x, j) for j in jac_layernorm_p]
    return tuple(jac_layernorm_p + jac_tlayer_ffn_p)

def t_gpt2_tlayer_sublock2_bkwd_x(layer_params, y, train=True): # input: seq_len x emb_dim
    y_diff = t_layernorm_fwd(layer_params[:2], y)
    jac_layernorm_x = t_layernorm_bkwd_x(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_ffn_fwd(layer_params[2:], y_diff, t_gelu_fwd), train)
    # TODO XXX: add dropout!
    jac_tlayer_ffn_x = t_tlayer_ffn_bkwd_x(layer_params[2:], y_diff, t_gelu_fwd)
    
    jac_y = torch.eye(y.numel(), device=y.device)    
    jac_y_diff = torch.einsum("abcdef, defghi->abcghi", jac_tlayer_ffn_x, jac_layernorm_x)
    return jac_y.reshape(jac_y_diff.shape) + jac_y_diff

def t_gpt2_tlayer_fwd(layer_params, y, mask, train=True): # input: seq_len x emb_dim
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train)
    y = t_gpt2_tlayer_sublock2_fwd(layer_params[-6:], y, train)
    return y

def t_gpt2_tlayer_bkwd_p(layer_params, y, mask, train=True): # input: seq_len x emb_dim
    jac_subblock1_p = t_gpt2_tlayer_sublock1_bkwd_p(layer_params[:-6], y, mask, train)
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train)
    jac_subblock2_p = t_gpt2_tlayer_sublock2_bkwd_p(layer_params[-6:], y, train)
    jac_subblock2_x = t_gpt2_tlayer_sublock2_bkwd_x(layer_params[-6:], y, train)
    
    jac_subblock1_p = _mult_jacs_in_2d(jac_subblock2_x, jac_subblock1_p, y)
    return tuple(jac_subblock1_p) + jac_subblock2_p

def t_gpt2_tlayer_bkwd_x(layer_params, y, mask, train=True): # input: seq_len x emb_dim
    jac_subblock1_x = t_gpt2_tlayer_sublock1_bkwd_x(layer_params[:-6], y, mask, train)
    y = t_gpt2_tlayer_sublock1_fwd(layer_params[:-6], y, mask, train)
    jac_subblock2_x = t_gpt2_tlayer_sublock2_bkwd_x(layer_params[-6:], y, train)
      
    return torch.einsum('abcdef, defghi->abcghi', jac_subblock2_x, jac_subblock1_x)

def t_gpt2_tlayers_fwd(params, y, mask, indices, train=True): # input: seq_len x
    y = t_embed_fwd(params[0], y)
    y = t_dropout(y + params[1][0], train)
    
    for layer_params in params[2:-1]:
        y = t_gpt2_tlayer_fwd(layer_params, y, mask, train)
    y = t_layernorm_fwd(params[-1], y)

    return y

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

def t_gpt2_tlayers_bkwd_p(params, y, mask, indices, train=True): # input: seq_len x
    indices = torch.arange(y.shape[1], device=y.device).unsqueeze(0).expand(*y.shape) # we ignore indices arg
    jac_embed = t_embed_bkwd(params[0], y)
    # Due to tying of embedding and final projection layers,
    # we need to fill zeroed gradient with respect to biases:
    jac_embed = [jac_embed[0], torch.zeros(jac_embed[0].shape[:-1], device=y.device)]
    y = t_embed_fwd(params[0], y)
    # Reuse t_embed_bkwd to compute jacobian of pos_encoding
    # Need to account for lack of  1/ sqrt(emb_dim)
    jac_pos_enc = t_embed_bkwd(params[1], indices) 
    jac_pos_enc = [jac_pos_enc[0] / params[1][0].shape[1] * 2, ]
    # TODO XXX: add dropout
    y = t_dropout(y + params[1][0], train)
    
    layers_jacs_p = []
    layers_jacs_x = []
    
    for layer_params in params[2:-1]:
        layers_jacs_p.append(t_gpt2_tlayer_bkwd_p(layer_params, y, mask, train))
        layers_jacs_x.append(t_gpt2_tlayer_bkwd_x(layer_params, y, mask, train))
        y = t_gpt2_tlayer_fwd(layer_params, y, mask, train)
    jac_layernorm_p = t_layernorm_bkwd_p(params[-1], y)
    jac_layernorm_x = t_layernorm_bkwd_x(params[-1], y)    
    y = t_layernorm_fwd(params[-1], y)
    
    # Propoagate back
    layers_jacs_x[-1]=torch.einsum('abcdef, defghi -> abcghi', jac_layernorm_x, layers_jacs_x[-1])
    layers_jacs_p[-1] = _mult_jacs_in_2d(jac_layernorm_x, layers_jacs_p[-1], y)
    for i in reversed(range(1, len(layers_jacs_p))):
        layers_jacs_x[i-1]=torch.einsum('abcdef, defghi -> abcghi',layers_jacs_x[i], layers_jacs_x[i-1])
        layers_jacs_p[i-1] = _mult_jacs_in_2d(layers_jacs_x[i], layers_jacs_p[i-1], y)
    jac_pos_enc[0] =torch.einsum('abcdef, defgh -> abcgh', layers_jacs_x[0], jac_pos_enc[0])
    jac_embed[0] = torch.einsum('abcdef, defgh -> abcgh', layers_jacs_x[0], jac_embed[0])

    return tuple([jac_embed, jac_pos_enc] + layers_jacs_p + [jac_layernorm_p])

def t_gpt2_forward(params, y, y_mask, y_indices, train): # input: seq_len x
    y = t_gpt2_tlayers_fwd(params, y, y_mask, y_indices, train=train)
    
    y = t_linear_fwd(params[0], y) 
    return y

def t_gpt2_bkwd_p(params, y, y_mask, y_indices, train): # input: seq_len x
    jac = t_gpt2_tlayers_bkwd_p(params, y, y_mask, y_indices, train=train)
    y = t_gpt2_tlayers_fwd(params, y, y_mask, y_indices, train=train)
    
    jac_linear_x = t_linear_bkwd_x(params[0], y) 
    jac_linear_p = t_linear_bkwd_p(params[0], y)    
    
    jac = list(jac)
    for i in range(len(jac)):
        jac[i] = _mult_jacs_in_2d(jac_linear_x, jac[i], y)
    
    # As we tie embedding and last projection weights
    jac[0] = (jac[0][0] + jac_linear_p[0], jac[0][1] + jac_linear_p[1])
    return tuple(jac)


#t_batched_forward_gpt2 = torch.vmap(t_forward_gpt2, in_dims=(None, 0, 0, 0, None), randomness="different") # TODO XXX: output will be batched unlike JAX's vmap
t_batched_forward_gpt2 = t_gpt2_forward # TODO XXX: rename the left one too
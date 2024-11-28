### PARAMS + MODEL
DROPOUT_RATE = 0.1 # TODO: move it out, and pass as paramteter

import math
import torch

### PARAMS: they are teh same as for Torch.Func

from model_torch_func import log_softmax, init_transformer_gpt2, count_num_params, batched_forward_gpt2

### MODEL in TRITON

def t_log_softmax(x_logits): # compute log_softmax from logits over the last dimension
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0] # as it returns (maxs, indices)
    return x_logits - torch.logsumexp(x_logits, axis=-1, keepdims=True)

def t_embed_fwd(layer_params, x): # input: 1 x
    return layer_params[0][x] * math.sqrt(layer_params[0].shape[1]) # since layer_params[0] is vocab_size x emb_dim

def t_embed_bkwd(layer_params, x): # input: 1 x
    emb_size = layer_params[0].shape[1]    
    fn_outdim = torch.numel(x) * emb_size
    fn_indim =  torch.numel(layer_params[0])
    jac = torch.zeros(fn_outdim, fn_indim, device=x.device)
    
    indices = torch.tile(torch.arange(emb_size, device=x.device), (x.numel(), 1))
    indices = ((x * emb_size).unsqueeze(1) + indices).reshape(-1, 1)
    jac.scatter_(1, indices, math.sqrt(emb_size))
    
    return (jac.reshape(torch.numel(x), emb_size, layer_params[0].shape[0], layer_params[0].shape[1]), )

def t_relu_fwd(x):
    return torch.where(torch.le(x, 0), 0, x) # inputs can broadcastable - follow pytorch's implementation

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

def t_proj_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params, -2, -1)) # since layer_params is ... x output_dim x emb_dim

def t_scaled_dot_prod_attn(qkv, mask, train=True): # inputs: batch_size x heads x 3 x seq_len x emb_dim, mask: batch_size x seq_len(q) x seq_len(k)
    q, k, v = torch.unbind(qkv, dim=2)# batch_size x heads x seq_len x emb_dim
    attn = torch.matmul(q, torch.transpose(k, -2, -1)) # seq_len(q) x seq_len(k)
    attn = attn / math.sqrt(q.shape[-1]) # scale by sqrt(d_k)
    #attn = jnp.where(mask, attn, jnp.full_like(attn, -jnp.inf))
    attn = torch.where(torch.unsqueeze(mask,dim=1), attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = torch.exp(log_softmax(attn))
    softmaxed_attn = t_dropout(softmaxed_attn, train)
    return torch.matmul(softmaxed_attn, v) # output: seq_len x emb_dim

def t_tlayer_attn_heads_fwd(layer_params, qkv, mask, train): # params: heads x 3 x emb_dim/heads x emb_dim, input: batch_size x seq_len x emb_dim
    qkv = torch.stack(qkv,dim=-3) # batch_size x 3 x seq_len x emb_dim
    
    proj_qkv = t_proj_fwd(layer_params, torch.unsqueeze(qkv, 1)) # batch_size x heads x 3 x seq_len x emb_dim
    return t_scaled_dot_prod_attn(proj_qkv, mask, train)

def t_tlayer_attn_fwd(layer_params, qkv, mask, train): # input: batch_size x seq_len x emb_dim
    num_heads = layer_params[0].shape[0]
    heads_attns = t_tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train)
    attn = torch.concatenate(torch.unbind(heads_attns, -3), axis=-1) # TODO XXX XXX: there is probably better way to go from [K, M, N] -> [M, K*N]. Or modify VMAP to return diff shape
    return t_proj_fwd(layer_params[-1], attn)

def t_tlayer_ffn_fwd(layer_params, x, activation_fn): # input: seq_len x emb_dim
    x = t_linear_fwd((layer_params[0], layer_params[1]), x)
    x = activation_fn(x)
    x = t_linear_fwd((layer_params[2], layer_params[3]), x)
    return x

def t_dropout(x, train=True):
    if not train: # As we jit the whole loss/inference, the train param is known at tracing time.
        return x * (1-DROPOUT_RATE)
    
    return x * torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE))

def t_layernorm_fwd(layer_params, x):
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims=True) # TODO XXX: Compute variance, add epsilon and take a sqrt instead (in order to avoid division by zero)
    normalized_x = (x - x_mean) / x_std
    return torch.multiply(x, layer_params[0][None, :]) + layer_params[1][None, :] # since both layer_params are output_dim x

def t_tlayer_fwd_gpt2(layer_params, y, mask, train=True): # input: seq_len x emb_dim

    y_diff = t_layernorm_fwd(layer_params[:2], y)
    y = y + t_dropout(t_tlayer_attn_fwd(layer_params[2:-6], (y_diff, y_diff, y_diff), mask, train), train)

    y_diff = t_layernorm_fwd(layer_params[-6:-4], y)
    y = y + t_dropout(t_tlayer_ffn_fwd(layer_params[-4:], y_diff, t_gelu_fwd), train)
    return y

def t_tlayers_fwd_gpt2(params, y, mask, indices, train=True): # input: seq_len x
    y = t_embed_fwd(params[0], y)
    y = t_dropout(y + params[1][0], train)
    
    for layer_params in params[2:-1]:
        y = t_tlayer_fwd_gpt2(layer_params, y, mask, train)
    y = t_layernorm_fwd(params[-1], y)

    return y

def t_forward_gpt2(params, y, y_mask, y_indices, train): # input: seq_len x
    y = t_tlayers_fwd_gpt2(params, y, y_mask, y_indices, train=train)
    
    y = t_linear_fwd(params[0], y) 
    return y


#t_batched_forward_gpt2 = torch.vmap(t_forward_gpt2, in_dims=(None, 0, 0, 0, None), randomness="different") # TODO XXX: output will be batched unlike JAX's vmap
t_batched_forward_gpt2 = t_forward_gpt2
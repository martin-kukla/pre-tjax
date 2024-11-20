### PARAMS + MODEL
DROPOUT_RATE = 0.1 # TODO: move it out, and pass as paramteter
INIT_SCALE = 2e-2 # In my previous AIYAIN experiment, I used 0.1. TODO XXX:  setup up Xavier/Glorot for AIYAIN instead?

import math
import torch

### PARAMS 

def init_linear_layer(m, n, scale=INIT_SCALE): 
    return scale * torch.randn((n, m), device="cuda"), torch.zeros((n,), device="cuda")

def init_proj_layer(emb_dim, proj_dim, scale=INIT_SCALE):
    return scale * torch.randn((proj_dim, emb_dim), device="cuda")

def init_layernorm_layer(n, scale=INIT_SCALE):
    return torch.ones((n, ), device="cuda"), torch.zeros((n,), device="cuda")

def init_tlayer_attn_heads(emb_dim, num_heads, scale=INIT_SCALE): 
    proj_dim = int(emb_dim / num_heads)
    return scale * torch.randn((num_heads, 3, proj_dim, emb_dim), device="cuda") # 3 for each of qkv

def init_tlayer_attn(emb_dim, num_heads, residual_scale=INIT_SCALE):
    return (init_tlayer_attn_heads(emb_dim, num_heads), ) + (init_proj_layer(emb_dim, emb_dim, residual_scale),)
    
def init_tlayer_ffn(emb_dim, ffn_dim, residual_scale=INIT_SCALE):
    return init_linear_layer(emb_dim, ffn_dim) +  init_linear_layer(ffn_dim, emb_dim, residual_scale)

def init_tlayer_gpt2(emb_dim, num_heads, ffn_dim, nlayers):
    residual_scale = INIT_SCALE/ math.sqrt(2*nlayers) # scaling of residual layers following the paper: there is some amgibuity which units should be affected..
    attns = init_layernorm_layer(emb_dim) + init_tlayer_attn(emb_dim, num_heads, residual_scale = residual_scale) 
    return attns + init_layernorm_layer(emb_dim) + init_tlayer_ffn(emb_dim, ffn_dim, residual_scale = residual_scale)

def init_transformer_gpt2(vocab_size, emb_dim, layers, num_heads, ffn_dim, seq_len):
    params = ( [init_linear_layer(emb_dim, vocab_size)] 
    + [(init_proj_layer(emb_dim, seq_len), )] # learnable positional encodings
    + [init_tlayer_gpt2(emb_dim, num_heads, ffn_dim, layers) for _ in range(layers)]
    + [init_layernorm_layer(emb_dim)])

    params = [list(p) for p in params]        
    return params

def count_num_params(params):
    return sum([sum([p.numel() for p in p_grp]) for p_grp in params])

### MODEL

def log_softmax(x_logits): # compute log_softmax from logits over the last dimension
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0] # as it returns (maxs, indices)
    return x_logits - torch.logsumexp(x_logits, axis=-1, keepdims=True)

def embed(layer_params, x): # input: 1 x
    return layer_params[0][x] * math.sqrt(layer_params[0].shape[1]) # since layer_params[0] is vocab_size x emb_dim

def relu(x):
    return torch.maximum(0, x)

def gelu(x):
    k = math.sqrt(2/math.pi)
    return 0.5 * x * (1 + torch.tanh(k * (x + 0.044715 * torch.pow(x,3))))

def linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params[0], 0, 1)) + layer_params[1][None, :] # since layer_params[0] is output_dim x emb_dim, layer_params[1] is output_dim

def proj_fwd(layer_params, x): # input: seq_len x emb_dim
    return torch.matmul(x, torch.transpose(layer_params, 0, 1)) # since layer_params is output_dim x emb_dim

def scaled_dot_prod_attn(qkv, mask, train=True): # inputs: seq_len x emb_dim, mask: seq_len(q) x seq_len(k)
    q, k, v = qkv
    attn = torch.matmul(q, torch.transpose(k, 0, 1)) # seq_len(q) x seq_len(k)
    attn = attn / math.sqrt(q.shape[-1]) # scale by sqrt(d_k)
    #attn = jnp.where(mask, attn, jnp.full_like(attn, -jnp.inf))
    attn = torch.where(mask, attn, torch.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = torch.exp(log_softmax(attn))
    softmaxed_attn = dropout(softmaxed_attn, train)
    return torch.matmul(softmaxed_attn, v) # output: seq_len x emb_dim

def tlayer_attn_head_fwd(layer_params, qkv, mask, train): # input: seq_len x emb_dim
    proj_qkv = tuple([proj_fwd(p, x) for p, x in zip(layer_params, qkv)]) #TODO: vmap? For cross attn, qkv are not of the same shape..
    return scaled_dot_prod_attn(proj_qkv, mask, train)

tlayer_attn_heads_fwd = torch.vmap(tlayer_attn_head_fwd, in_dims=(0, None, None, None), randomness="different")

def tlayer_attn_fwd(layer_params, qkv, mask, train): # input: seq_len x emb_dim
    num_heads = layer_params[0].shape[0]
    heads_attns = tlayer_attn_heads_fwd(layer_params[0], qkv, mask, train)
    attn = torch.concatenate(torch.unbind(heads_attns, 0), axis=-1) # TODO XXX: there is probably better way to go from [K, M, N] -> [M, K*N]. Or modify VMAP to return diff shape
    return proj_fwd(layer_params[-1], attn)

def tlayer_ffn_fwd(layer_params, x, activation_fn): # input: seq_len x emb_dim
    x = linear_fwd((layer_params[0], layer_params[1]), x)
    x = activation_fn(x)
    x = linear_fwd((layer_params[2], layer_params[3]), x)
    return x

def dropout(x, train=True):
    if not train: # As we jit the whole loss/inference, the train param is known at tracing time.
        return x * (1-DROPOUT_RATE)
    
    return x * torch.bernoulli(torch.full_like(x, 1-DROPOUT_RATE))

def layernorm_fwd(layer_params, x):
    x_mean = torch.mean(x, axis=-1, keepdims=True)
    x_std = torch.std(x, axis=-1, keepdims=True) # TODO XXX: Compute variance, add epsilon and take a sqrt instead (in order to avoid division by zero)
    normalized_x = (x - x_mean) / x_std
    return torch.multiply(x, layer_params[0][None, :]) + layer_params[1][None, :] # since both layer_params are output_dim x

def tlayer_fwd_gpt2(layer_params, y, mask, train=True): # input: seq_len x emb_dim

    y_diff = layernorm_fwd(layer_params[:2], y)
    y = y + dropout(tlayer_attn_fwd(layer_params[2:-6], (y_diff, y_diff, y_diff), mask, train), train)

    y_diff = layernorm_fwd(layer_params[-6:-4], y)
    y = y + dropout(tlayer_ffn_fwd(layer_params[-4:], y_diff, gelu), train)
    return y

def tlayers_fwd_gpt2(params, y, mask, indices, train=True): # input: seq_len x
    y = embed(params[0], y)
    y = dropout(y + params[1][0], train)
    
    for layer_params in params[2:-1]:
        y = tlayer_fwd_gpt2(layer_params, y, mask, train)
    y = layernorm_fwd(params[-1], y)

    return y

def forward_gpt2(params, y, y_mask, y_indices, train): # input: seq_len x
    y = tlayers_fwd_gpt2(params, y, y_mask, y_indices, train=train)
    
    y = linear_fwd(params[0], y) 
    return y

batched_forward_gpt2 = torch.vmap(forward_gpt2, in_dims=(None, 0, 0, 0, None), randomness="different") # TODO XXX: output will be batched unlike JAX's vmap
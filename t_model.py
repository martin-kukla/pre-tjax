### PARAMS + MODEL
DROPOUT_RATE = 0.1 # TODO: move it out, and pass as paramteter

import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp

### PARAMS 

def init_linear_layer(m, n, key, scale=1e-2): 
    w_key, b_key = random.split(key) 
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_proj_layer(emb_dim, proj_dim, key, scale=1e-2): 
    return scale * random.normal(key, (proj_dim, emb_dim))

def init_tlayer_attn_heads(emb_dim, num_heads, key, scale=1e-2): 
    proj_dim = int(emb_dim / num_heads)
    return scale * random.normal(key, (num_heads, 3, proj_dim, emb_dim)) # 3 for each of qkv

def init_tlayer_attn(emb_dim, num_heads, key):
    if num_heads>0:
        keys = random.split(key, 2)
        return (init_tlayer_attn_heads(emb_dim, num_heads, keys[0]), ) + (init_proj_layer(emb_dim, emb_dim, keys[1]),)
    else:
        return tuple() # TODO: head=0 means no projections, I leave this option for further investigation
    
def init_tlayer_ffn(emb_dim, ffn_dim, key):
    keys = random.split(key, 2)
    return init_linear_layer(emb_dim, ffn_dim, keys[0]) +  init_linear_layer(ffn_dim, emb_dim, keys[1]) 

def init_tlayer(emb_dim, num_heads, ffn_dim, key, cross_attn=False):
    keys = random.split(key, 3 if cross_attn else 2)
    attns = init_tlayer_attn(emb_dim, num_heads, keys[0])
    if cross_attn:
        attns = attns + init_tlayer_attn(emb_dim, num_heads, keys[2])
    return attns + init_tlayer_ffn(emb_dim, ffn_dim, keys[1])


### MODEL

def log_softmax(x_logits): # compute log_softmax from logits over the last dimension
    return x_logits - logsumexp(x_logits, axis=-1, keepdims=True)

def embed(layer_params, x, d_model): # input: 1 x
    return layer_params[0][x] * jnp.sqrt(d_model) # since layer_params[0] is vocab_size x emb_dim

def relu(x):
    return jnp.maximum(0, x)

def linear_fwd(layer_params, x): # input: seq_len x emb_dim
    return jnp.matmul(x, jnp.transpose(layer_params[0])) + layer_params[1][None, :] # since layer_params[0] is output_dim x emb_dim, layer_params[1] is output_dim

def proj_fwd(layer_params, x): # input: seq_len x emb_dim
    return jnp.matmul(x, jnp.transpose(layer_params)) # since layer_params is output_dim x emb_dim

def scaled_dot_prod_attn(qkv, mask): # inputs: seq_len x emb_dim, mask: seq_len(q) x seq_len(k)
    q, k, v = qkv
    attn = jnp.matmul(q, jnp.transpose(k)) # seq_len(q) x seq_len(k)
    attn = attn / jnp.sqrt(q.shape[-1]) # scale by sqrt(d_k)
    #attn = jnp.where(mask, attn, jnp.full_like(attn, -jnp.inf))
    attn = jnp.where(mask, attn, jnp.full_like(attn, -1e9)) # Note, instead of usign -jnp.inf, which results in NaNs (NIT: probably better to use jax.numpy.finfo)
    softmaxed_attn = jnp.exp(log_softmax(attn))
    return jnp.matmul(softmaxed_attn, v) # output: seq_len x emb_dim

def tlayer_attn_head_fwd(layer_params, qkv, mask): # input: seq_len x emb_dim
    proj_qkv = tuple([proj_fwd(p, x) for p, x in zip(layer_params, qkv)]) #TODO: vmap? For cross attn, qkv are not of the same shape..
    return scaled_dot_prod_attn(proj_qkv, mask)

tlayer_attn_heads_fwd = vmap(tlayer_attn_head_fwd, in_axes=(0, None, None))

def tlayer_attn_fwd(layer_params, qkv, mask): # input: seq_len x emb_dim
    # TODO: why with projection learning is slower/worse (update: validate that it's worse)?
    # NOTE, this is a jit hack: if is architectural i.e. it doens't change across execution, so it should not cause issues for lax'tracing
    if len(layer_params) > 0: 
        num_heads = int((len(layer_params)-1)/3)
        heads_attns = tlayer_attn_heads_fwd(layer_params[0], qkv, mask)
        attn = jnp.concatenate(heads_attns, axis=-1)
        return proj_fwd(layer_params[-1], attn)
    else:
        return scaled_dot_prod_attn(qkv, mask)

def tlayer_ffn_fwd(layer_params, x): # input: seq_len x emb_dim
    x = linear_fwd((layer_params[0], layer_params[1]), x)
    x = relu(x)
    x = linear_fwd((layer_params[2], layer_params[3]), x)
    return x

def dropout(x, key, train=True):
    if not train: # As we jit the whole loss/inference, the train param is known at tracing time.
        return x * (1-DROPOUT_RATE)
    
    return x * random.bernoulli(key, 1-DROPOUT_RATE, x.shape)

def layernorm(x):
    x_mean = jnp.mean(x, axis=-1, keepdims=True)
    x_std = jnp.std(x, axis=-1, keepdims=True)
    return (x - x_mean) / x_std

def tlayer_fwd(layer_params, y, mask, key, train=True): # input: seq_len x emb_dim
    keys = random.split(key, 2)

    y = y + dropout(tlayer_attn_fwd(layer_params[:-4], (y, y, y), mask), keys[0], train)
    y = layernorm(y)

    y = y + dropout(tlayer_ffn_fwd(layer_params[-4:], y), keys[1], train)
    y = layernorm(y)
    return y

def tlayer_with_cross_attn_fwd(layer_params, y, mask, x, yx_mask, key, train=True): # input: seq_len x emb_dim
    cross_attn_section = int((len(layer_params) - 4)/2)
    keys = random.split(key, 3)
    y = y + dropout(tlayer_attn_fwd(layer_params[:cross_attn_section], (y, y, y), mask), keys[0], train)
    y = layernorm(y)

    #cross attn
    y = y + dropout(tlayer_attn_fwd(layer_params[cross_attn_section:-4], (y, x, x), yx_mask), keys[1], train)
    y = layernorm(y)
    
    y = y + dropout(tlayer_ffn_fwd(layer_params[-4:], y), keys[2], train)
    y = layernorm(y)
    return y

def pos_encodings(x): # input: seq_len x emb_dim
    seq_len, emb_dim = x.shape

    indices = jnp.arange(seq_len)[:, None] 
    div_term = jnp.fromfunction(lambda i: 1 / pow(10000, 2 * i/emb_dim), (int(emb_dim/2),), dtype=float)[None, :]
    pos_array = jnp.dot(indices, div_term)
    return jnp.stack((jnp.sin(pos_array), jnp.cos(pos_array)), axis=2).reshape(seq_len, emb_dim)
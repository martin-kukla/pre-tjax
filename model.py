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

def init_layernorm_layer(n, key, scale=1e-2): 
    #w_key, b_key = random.split(key) 
    #return scale * random.normal(w_key, (n, )), scale * random.normal(b_key, (n,))
    return jnp.ones((n, )), jnp.zeros((n,))

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
    keys = random.split(key, 6 if cross_attn else 4)
    attns = init_tlayer_attn(emb_dim, num_heads, keys[0]) + init_layernorm_layer(emb_dim, keys[1])
    if cross_attn:
        attns = attns + init_tlayer_attn(emb_dim, num_heads, keys[2]) + init_layernorm_layer(emb_dim, keys[3])
    return attns + init_tlayer_ffn(emb_dim, ffn_dim, keys[-2]) + init_layernorm_layer(emb_dim, keys[-1])

def init_transformer(vocab_size, emb_dim, layers, num_heads, ffn_dim, key): 
    all_keys = random.split(key, 2 * layers + 1)

    encoder_params = [init_tlayer(emb_dim, num_heads, ffn_dim, k) for k in all_keys[1:1+layers]] 
    #reshaped_encoder_params = [jnp.stack(p_grp) for p_grp in zip(*encoder_params)]

    decoder_params = [init_tlayer(emb_dim, num_heads, ffn_dim, k, cross_attn=True) for k in all_keys[1+layers:]]
    #reshaped_decoder_params = [jnp.stack(p_grp) for p_grp in zip(*decoder_params)]
    
    params = ( [init_linear_layer(emb_dim, vocab_size, all_keys[0])] 
    # Using scan for tlayers is not computationally efficient on GPU: check https://github.com/google/jax/discussions/16106#discussioncomment-5992623
    #+ [reshaped_encoder_params] 
    + encoder_params
    #+ [reshaped_decoder_params]) 
    + decoder_params)

    params = [list(p) for p in params]        
    return params

### MODEL

def log_softmax(x_logits): # compute log_softmax from logits over the last dimension
    return x_logits - logsumexp(x_logits, axis=-1, keepdims=True)

def embed(layer_params, x): # input: 1 x
    return layer_params[0][x] * jnp.sqrt(layer_params[0].shape[1]) # since layer_params[0] is vocab_size x emb_dim

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

def layernorm_fwd(layer_params, x):
    x_mean = jnp.mean(x, axis=-1, keepdims=True)
    x_std = jnp.std(x, axis=-1, keepdims=True) # TODO XXX: Compute variance, add epsilon and take a sqrt instead (in order to avoid division by zero)
    normalized_x = (x - x_mean) / x_std
    return jnp.multiply(x, layer_params[0][None, :]) + layer_params[1][None, :] # since both layer_params are output_dim x

def tlayer_fwd(layer_params, y, mask, key, train=True): # input: seq_len x emb_dim
    keys = random.split(key, 2)

    y = y + dropout(tlayer_attn_fwd(layer_params[:-8], (y, y, y), mask), keys[0], train)
    y = layernorm_fwd(layer_params[-8:-6], y)

    y = y + dropout(tlayer_ffn_fwd(layer_params[-6:-2], y), keys[1], train)
    y = layernorm_fwd(layer_params[-2:], y)
    return y

def tlayer_with_cross_attn_fwd(layer_params, y, mask, x, yx_mask, key, train=True): # input: seq_len x emb_dim
    cross_attn_section = int((len(layer_params) - 6)/2)
    keys = random.split(key, 3)
    y = y + dropout(tlayer_attn_fwd(layer_params[:cross_attn_section-2], (y, y, y), mask), keys[0], train)
    y = layernorm_fwd(layer_params[cross_attn_section-2:cross_attn_section], y)

    #cross attn
    y = y + dropout(tlayer_attn_fwd(layer_params[cross_attn_section:-8], (y, x, x), yx_mask), keys[1], train)
    y = layernorm_fwd(layer_params[-8:-6], y)
    
    y = y + dropout(tlayer_ffn_fwd(layer_params[-6:-2], y), keys[2], train)
    y = layernorm_fwd(layer_params[-2:], y)
    return y

def pos_encodings(x, indices): # input: seq_len x emb_dim
    seq_len, emb_dim = x.shape

    #indices = jnp.arange(seq_len)[:, None] 
    indices = indices[:, None] 
    div_term = jnp.fromfunction(lambda i: 1 / pow(10000, 2 * i/emb_dim), (int(emb_dim/2),), dtype=float)[None, :]
    pos_array = jnp.dot(indices, div_term)
    return jnp.stack((jnp.sin(pos_array), jnp.cos(pos_array)), axis=2).reshape(seq_len, emb_dim)

# TODO: unify tlayer(s) with tlayer(s)_with_cross_attention, but that jit works correctly
def tlayers_fwd(params, y, mask, indices, key, train=True): # input: seq_len x
    key, dropout_key = random.split(key, 2)
    y = embed(params[0], y)
    y = dropout(y + pos_encodings(y, indices), dropout_key, train)
    
    for layer_params in params[1:]:
        key, tlayer_key = random.split(key, 2)
        y = tlayer_fwd(layer_params, y, mask, tlayer_key, train)

    return y

def tlayers_fwd_scanned(params, y, mask, key, train=True): # input: seq_len x
    key, dropout_key = random.split(key, 2)
    y = embed(params[0], y)
    y = dropout(y + pos_encodings(y), dropout_key, train)
    
    def tlayer_fwd_scan_step(y_and_key, tlayer_params):
        y, key = y_and_key
        key, tlayer_key = random.split(key, 2)
        y = tlayer_fwd(tlayer_params, y, mask, tlayer_key, train)
        return (y, key), None

    y_and_key, _ = lax.scan(tlayer_fwd_scan_step, (y, key), params[1]) 
    
    return y_and_key[0]

def tlayers_with_cross_attn_fwd(params, y, mask, x, yx_mask, y_indices, key, train=True): # input: seq_len x
    keys = random.split(key, len(params) )
    y = embed(params[0], y)
    y = dropout(y + pos_encodings(y, y_indices), keys[0], train)
    
    for layer_params, layer_key in zip(params[1:], keys[1:]):
        y = tlayer_with_cross_attn_fwd(layer_params, y, mask, x, yx_mask, layer_key, train)
    return y

def tlayers_with_cross_attn_fwd_scanned(params, y, mask, x, yx_mask, key, train=True): # input: seq_len x
    key, dropout_key = random.split(key, 2)
    y = embed(params[0], y)
    y = dropout(y + pos_encodings(y), dropout_key, train)

    def tlayer_fwd_scan_step(y_and_key, tlayer_params):
        y, key = y_and_key
        key, tlayer_key = random.split(key, 2)
        y = tlayer_with_cross_attn_fwd(tlayer_params, y, mask, x, yx_mask, tlayer_key, train)
        return (y, key), None

    y_and_key, _ = lax.scan(tlayer_fwd_scan_step, (y, key), params[1]) 
    
    return y_and_key[0]

def forward(params, x, y, x_mask, y_mask, yx_mask, x_indices, y_indices, key, train): # input: seq_len x
    layers = int((len(params) -1) /2)
    keys = random.split(key, 2)
    
    x = tlayers_fwd(params[:layers+1], x, x_mask, x_indices, keys[0], train=train)
    #x = tlayers_fwd_scanned(params[:2], x, x_mask, keys[0], train=train)
    
    y = tlayers_with_cross_attn_fwd([params[0]] + params[layers+1:], y, y_mask, x, yx_mask, y_indices, keys[1], train=train)
    #y = tlayers_with_cross_attn_fwd_scanned([params[0], params[2]], y, y_mask, x, yx_mask, keys[1], train=train)
    y = linear_fwd(params[0], y) 
    return y

#batched_forward = jit(vmap(forward, in_axes=(None, 0, 0, None, None)), static_argnames=['train'])
batched_forward = vmap(forward, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None, None))
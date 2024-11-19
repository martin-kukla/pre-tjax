###
# Loss + grads
###

import jax
from functools import partial
import jax.numpy as jnp
from model import log_softmax, batched_forward_gpt2 # TODO XXX XXX: pass forward fn as parameter to relevant functions instead!
from jax import grad, jit, lax 
from jax import random

def avg_cross_entropy_loss(y_labels, x_logits): # y_labels: batch_len x seq_len, x_logits: batch_len x seq_len x vocab_size
    # Note that in jax, un-jitted reshape calls are producing copies of array instead of views.
    # However, for jitted, this SHOULD be optmized away (I checked this function that indeed it is).
    y_labels_1d = jnp.reshape(y_labels, -1) # there is probably a way of doing it while staying in 2d..
    x_logits_2d = jnp.reshape(x_logits, (y_labels.size, -1))
    elements_loss = log_softmax(x_logits_2d)[(jnp.arange(y_labels.size), y_labels_1d)]
    elements_loss = jnp.where(y_labels_1d != 0, elements_loss, jnp.nan) # account for padding tokens
    result = -jnp.nanmean(elements_loss) 
    return result, jnp.count_nonzero(y_labels)
    
def accuracy(y_labels, x_logits):
    return jnp.nanmean(jnp.where(y_labels!=0, y_labels == jnp.argmax(x_logits, axis=-1), jnp.nan))

@partial(jax.jit, static_argnames=['sample_len', 'start_tok', 'end_tok']) # TODO XXX: don't pass y_mask nor y_indices (pass batch_size though!)
def predict(params, y_mask, y_indices, sample_len, start_tok, end_tok): # TODO: code up not-scanned version, which could be faster on GPU
    def predict_step(step_i, y):
        # TODO: Cache key-value pairs
        new_y = batched_forward_gpt2(params, y, y_mask, y_indices, random.PRNGKey(0), False) 
        new_toks = jnp.argmax(new_y[:, step_i], axis=-1)
        y = y.at[:,step_i+1].set(new_toks)
        return y
    
    start_toks = jnp.full((y_mask.shape[0], sample_len), start_tok)
    y_sample = jax.lax.fori_loop(0, sample_len, predict_step, start_toks) 
    y_sample = jnp.where(jax.lax.cummax(y_sample, axis=1) != end_tok, y_sample, 0) # replace END token, and what follows with padding

    y_sample = y_sample[:, 1:]
    return jnp.where(y_sample!=start_tok, y_sample, 0) # It should not be happening, but for random model it might.2

def loss(params, y, y_mask, y_indices, key, train):  # inputs: batch_size x seq_len
    y_in = y[:, :-1]
    y_out = y[:, 1:]
    
    # TODO: write it without copying memory? is it possible? 
    logits = batched_forward_gpt2(params, y_in, y_mask, y_indices, key, train) 
    loss_val, tokens_count = avg_cross_entropy_loss(y_out, logits)
    acc = accuracy(y_out, logits) # TODO: Do I need to stop_gradient on this? I think not, but double-check
    return loss_val, (loss_val, acc, tokens_count/jnp.size(y_out)) # TODO: this is wrapping, but we could make use of jax.value_and_grad instead

loss_train = partial(loss, train=True)
loss_eval = jit(partial(loss, key=random.PRNGKey(0), train=False))

grad_loss = jit(grad(loss_train, has_aux=True))
#grad_loss = grad(loss_train, has_aux=True)

#print(f'iter #{i} loss {loss_train(params, jnp.array(x[:1]), jnp.array(y[:1]), random.PRNGKey(0))[0] }')

#with jax.disable_jit():
#print(f'iter #{i} loss {predict(params, jnp.array(x[:2], 50, START_TOK, END_TOK))}')

# TODO XXX XXX: write some test for it?
@jit #TODO XXX: take y_prefix_len which not to ignore probs for?
def log_probs(params, y, y_mask, y_indices):  # inputs: batch_size x seq_len
    y_in = y[:, :-1]
    y_out = y[:, 1:]

    # copied a few lines from avg_cross_entropy_loss # TODO XXX XXX: reuse instead!
    def compute_log_probs(y_labels, x_logits): # y_labels: batch_len x seq_len, x_logits: batch_len x seq_len x vocab_size
        y_labels_1d = jnp.reshape(y_labels, -1) # there is probably a way of doing it while staying in 2d..
        x_logits_2d = jnp.reshape(x_logits, (y_labels.size, -1))
        elements_loss = log_softmax(x_logits_2d)[(jnp.arange(y_labels.size), y_labels_1d)]
        elements_loss = jnp.where(y_labels_1d != 0, elements_loss, 1) # account for padding tokens
        elements_loss_2d = jnp.reshape(elements_loss, (x_logits.shape[0], x_logits.shape[1]))
        y_log_probs = jnp.sum(elements_loss_2d, axis=1)
        return y_log_probs
    
    # TODO: write it without copying memory? is it possible? 
    logits = batched_forward_gpt2(params, y_in, y_mask, y_indices, random.PRNGKey(0), False) 
    return compute_log_probs(y_out, logits)


### 
# Optimizers
###

# TODO: any call to this function can be replaced by jax's tree_map
def elwise(params_and_grads, func): # generically applying func element-wise
    return [ [ func(*p_and_g) for p_and_g in zip(*p_and_g_items)] for p_and_g_items in zip(*params_and_grads)]

def sgd(params, grads, lr):
    return elwise((params, grads), lambda p,g: p - lr * g)


def init_adam_w(params):
    return [elwise((params,), lambda p: jnp.zeros_like(p)) for _ in range(2)]

@jit
def adam_w(params, grads, lr, betas, epsilon, moments, i, weight_decay=0.0): #TODO: add implemntation of weight_decay_mask?
    t = i + 1 # TODO: should we decuple iteration from t, and threading t instead?
    moments = [elwise((moment, grads), lambda m, g: b*m + (1-b) * pow(g, pow_g)) for b, moment, pow_g in zip(betas, moments, [1,2])]
    bias_corrected_moments = [elwise((moment,), lambda m: m / (1 - pow(b,t))) for b, moment in zip(betas, moments)]
    params = elwise((params, *bias_corrected_moments), lambda p,m,v: p - lr *(m / (jnp.sqrt(v) + epsilon) + weight_decay * p))
    return params, moments

from functools import partial
@partial(jax.jit, donate_argnames=("params","moments"), static_argnames=["weight_decay_mask"])
def adam_w_in_place(params, grads, lr, betas, epsilon, moments, i, weight_decay=0.0, weight_decay_mask=None):
    t = i + 1 # TODO: should we decuple iteration from t, and threading t instead?

    # TODO: once write it more effiently, combine both loops + vmap (if possible)?
    
    # update moments
    for b, moment, pow_g in zip(betas, moments, [1,2]): 
        for grp_i in range(len(grads)):
            for p_i in range(len(grads[grp_i])):
                moment[grp_i][p_i] = moment[grp_i][p_i].at[:].multiply(b)
                moment[grp_i][p_i] = moment[grp_i][p_i].at[:].add((1-b) * pow(grads[grp_i][p_i], pow_g))

    # update grads
    for grp_i in range(len(grads)):
        for p_i in range(len(grads[grp_i])):
            bias_correct_func = lambda b, m: m / (1 - pow(b,t))
            m = bias_correct_func(betas[0], moments[0][grp_i][p_i])
            v = bias_correct_func(betas[1], moments[1][grp_i][p_i])
            item_weight_decay = weight_decay if (weight_decay_mask is None or weight_decay_mask[grp_i][p_i]) else 0.0
            #item_weight_decay = weight_decay if weight_decay_mask[grp_i][p_i] else 0.0
            #print(f'Shape {grp_i}, {p_i}: {params[grp_i][p_i].shape} {item_weight_decay}')
            params[grp_i][p_i] =  params[grp_i][p_i].at[:].add(-lr * (m / (jnp.sqrt(v) + epsilon) + item_weight_decay * params[grp_i][p_i])) # TODO XXX: Test
            
    return params, moments

# Testing Adam in place:
#original_pointer = params[0][0].unsafe_buffer_pointer()
#params, moments = adam_w_in_place(params, grads, lr, betas, epsilon, moments, i)
#assert params[0][0].unsafe_buffer_pointer() == original_pointer # will not fail
#params, moments = adam_w(params, grads, lr, betas, epsilon, moments, i)
#assert params[0][0].unsafe_buffer_pointer() == original_pointer # will fail
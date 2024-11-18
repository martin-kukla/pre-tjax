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

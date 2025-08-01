# This currently contains Loss+Optimizer logic for both
# frameworks: Torch.Func and Triton.
# The function for the latter are prefixed with t_
# Some of the functions are shared between two.
# TODO: We should probably separate the logic into two separate files.

# ##
# Loss + grads
# ##

from functools import partial
import math
import torch
import triton
import triton.language as tl
from torch.func import grad
from model_torch_func import log_softmax, batched_forward_gpt2
from model_triton import t_log_softmax_fwd, t_log_softmax_bkwd, t_log_softmax_bkwd2, t_gpt2_forward, t_gpt2_forward_with_acts, t_gpt2_forward_with_acts_t, t_gpt2_bkwd_p, t_gpt2_bkwd2_p, t_gpt2_bkwd3_p, t_gpt2_bkwd3_p_t, _mult_jacs_in_2d

def avg_cross_entropy_loss(y_labels, x_logits):
    return _avg_cross_entropy_loss(log_softmax, y_labels, x_logits)

def t_avg_cross_entropy_loss(y_labels, x_logits):
    return _avg_cross_entropy_loss(t_log_softmax_fwd, y_labels, x_logits)

def _avg_cross_entropy_loss(log_softmax_fn, y_labels, x_logits): # y_labels: BS x N, x_logits: BS x S x vocab_size
    y_labels_1d = y_labels.reshape((-1,)) 
    x_logits_2d = x_logits.reshape((y_labels.numel(), -1))
    elements_loss = log_softmax_fn(x_logits_2d)[(torch.arange(y_labels.numel()), y_labels_1d)]
    elements_loss = torch.where(y_labels_1d != 0, elements_loss, float('nan'))
    result = -torch.nanmean(elements_loss) 
    return result, torch.count_nonzero(y_labels)

def t_avg_cross_entropy_loss_bkwd(y_labels, x_logits):
    y_labels_1d = y_labels.reshape((-1,))
    x_logits_2d = x_logits.reshape((y_labels.numel(), -1))
    elements_loss = t_log_softmax_fwd(x_logits_2d)[(torch.arange(y_labels.numel()), y_labels_1d)]
    elements_loss = torch.where(y_labels_1d != 0, elements_loss, float('nan'))
    
    jac_softmax = t_log_softmax_bkwd(x_logits_2d)[(torch.arange(y_labels.numel()), y_labels_1d)]
    # TODO XXX: code up derivative for torch.nanmean
    jac_nanmean = -torch.func.jacrev(torch.nanmean)(elements_loss) 
    jac_x_logits = torch.einsum("a, abc -> bc", jac_nanmean, jac_softmax)
    
    return jac_x_logits.reshape(x_logits.shape)

def t_avg_cross_entropy_loss_bkwd2(y_labels, x_logits):
    y_labels_1d = y_labels.reshape((-1,))
    x_logits_2d = x_logits.reshape((y_labels.numel(), -1))
    elements_loss = t_log_softmax_fwd(x_logits_2d)[(torch.arange(y_labels.numel()), y_labels_1d)]
    elements_loss = torch.where(y_labels_1d != 0, elements_loss, float('nan'))
    
    # propagate back
    # TODO XXX: code up derivative for torch.nanmean 
    jac_nanmean = -torch.func.jacrev(torch.nanmean)(elements_loss) 
    dloss_dx = torch.zeros_like(x_logits_2d) # bkwd for indexing
    y_labels_1d = y_labels_1d.to(torch.int64) # HOTFIX. TODO XXX: Think whether we shouldn't pass ys in int64 instead?
    dloss_dx.scatter_(1, y_labels_1d.unsqueeze(1), jac_nanmean.unsqueeze(1))
    
    # "Fused" t_log_softmax_bkwd2 with t_avg_cross_entropy_loss_bkwd2:
    # t_log_softmax_bkwd2 alone would create (BS, V, V) tensor, where V is vocabulary size.
    # By swapping order of ops, we don't need to create full (BS, V, V).
    # This only applies in context of avg_cross_entopy_loss_bkwd2, as dloss_dx
    # has 1-of-N encodings as vectors.
    def fused_t_log_softmax_bkwd2(dloss_dx, y_labels_1d, jac_nanmean, x_logits):
        BS, N = x_logits.shape

        x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0]
        logsums = torch.logsumexp(x_logits, axis=-1, keepdims=True)
        exp_logsums = torch.exp(logsums) # Q: is it going to be numerically stable?

        n_jac = -torch.exp(x_logits)/exp_logsums
        dloss_dx.add_(jac_nanmean.unsqueeze(1) * n_jac)
        return dloss_dx
    #dloss_dx = t_log_softmax_bkwd2(dloss_dx, x_logits_2d)
    dloss_dx = fused_t_log_softmax_bkwd2(dloss_dx, y_labels_1d, jac_nanmean, x_logits_2d)
    
    return dloss_dx.reshape(x_logits.shape)

# Note, this doesn't follow the convention of _bkwd3 functions:
# It's the last op of the graph, thus we fuse fwd and bkwd passes.
# We return the results of both passes
def t_avg_cross_entropy_loss_bkwd3(y_labels, x_logits):
    dloss_dx_shape = x_logits.shape 
    y_labels = y_labels.reshape((-1,1))
    y_labels = y_labels.to(torch.int64) # TODO XXX: shouldn't we pass y_labels in int64 already?
    x_logits = x_logits.reshape((y_labels.numel(), -1))
    nonzero_count = torch.count_nonzero(y_labels)
    
    # Modified log_softmax_fn with swapped the order of ops (indexing<->subtraction),
    # and computes values which are reused in propagation below
    x_logits = x_logits - torch.max(x_logits, axis=-1, keepdims=True)[0]
    x_logits_logsumexp = torch.logsumexp(x_logits, axis=-1, keepdims=True)
    x_logits_indexed = torch.gather(x_logits, 1, y_labels)
    elements_loss = torch.where(y_labels != 0, x_logits_indexed - x_logits_logsumexp, float('nan'))
    loss = -torch.nanmean(elements_loss) 
    
    # propagate back
    jac_nanmean = torch.where(y_labels != 0, -1/nonzero_count, 0)
    x_logits_sumexp = torch.exp(x_logits_logsumexp) # Q: is it going to be numerically stable?
    log_softmax_jac = -torch.exp(x_logits)/x_logits_sumexp # note we can precopmute exp(x_logits) as part of logsumpexp before   
    # TODO T: below I still need to create another array, can't src just be "1"?
    log_softmax_jac.scatter_add_(1, y_labels, torch.ones_like(log_softmax_jac)) # bkwd for indexing
    dloss_dx = jac_nanmean * log_softmax_jac
    
    return loss, nonzero_count, dloss_dx.reshape(dloss_dx_shape)

@triton.jit
def t_avg_cross_entropy_loss_bkwd3_k(y_labels_ptr,
                    x_logits_ptr,
                    loss_ptr,
                    dloss_dx_ptr,
                    aux_idx_ptr, # Auxilary indexing operation, as alternativ for tl.gather. It uses global memory: overhead + slow
                    x_logits_row_stride,
                    dloss_dx_row_stride,
                    aux_idx_row_stride,
                    nonzero_count_ptr,
                    n_rows,
                    n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    num_stages: tl.constexpr,
                    ):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    blcks = tl.cdiv(n_cols, BLOCK_SIZE)
    nonzero_count = tl.load(nonzero_count_ptr)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages):
        y_label = tl.load(y_labels_ptr + row_idx) # TODO T: load once, and keep it in shared memory?
        x_logits_row_start_ptr = x_logits_ptr + row_idx * x_logits_row_stride
        
        # Online softmax (https://arxiv.org/pdf/1805.02867)
        # computes x_logits_max & x_logits_sumexp with one memory load 
        x_logits_max = -1e9
        x_logits_sumexp = 0.0
        d_s = tl.full((BLOCK_SIZE, ), 1, dtype=tl.float32)
        for blck_idx in tl.range(0, blcks):
            offsets = blck_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x_logits = tl.load(x_logits_row_start_ptr + offsets, mask=mask, other=-1e9)
            blck_x_logits_max = tl.max(x_logits, axis=0)
            n_x_logits_max = tl.maximum(x_logits_max, blck_x_logits_max)
            x_logits_sumexp = x_logits_sumexp * tl.exp(x_logits_max - n_x_logits_max) + tl.sum(tl.exp(x_logits - n_x_logits_max))
            x_logits_max = n_x_logits_max
        
        # If not padding token, contribute to loss/dloss_dx computation
        if y_label!=0:
            for blck_idx in tl.range(0, blcks):
                offsets = blck_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_cols
                x_logits = tl.load(x_logits_row_start_ptr + offsets, mask=mask, other=-1e9)
                x_logits = x_logits - x_logits_max
                x_logits_exp = tl.exp(x_logits)
                
                if y_label>= blck_idx*BLOCK_SIZE and y_label<(blck_idx+1)*BLOCK_SIZE:    
                    # Workaround for the lack of tl.gather for now (slow, as it uses non-local memory)
                    #loss = tl.gather(x_logits, y_label, 0) - x_logits_logsumexp
                    aux_idx_row_start_ptr = aux_idx_ptr + row_idx * aux_idx_row_stride
                    aux_idx_offsets = tl.arange(0, BLOCK_SIZE)
                    aux_idx = tl.load(aux_idx_row_start_ptr + aux_idx_offsets) # no need for mask here
                    logit_for_y = tl.sum(x_logits * aux_idx)
                    loss = logit_for_y - tl.log(x_logits_sumexp)
                    loss = - loss/nonzero_count
                    tl.atomic_add(loss_ptr, loss)

                # propagate back:
                # I used torch.scatter_add_, which I am not sure I can perform locally? thus
                # using atomic_add twice
                dloss_dx_row_start_ptr = dloss_dx_ptr + row_idx * dloss_dx_row_stride
                if y_label>= blck_idx*BLOCK_SIZE and y_label<(blck_idx+1)*BLOCK_SIZE: # TODO T: Can I do it using mask?
                    tl.atomic_add(dloss_dx_row_start_ptr + y_label, -1/nonzero_count)
                log_softmax_jac = - x_logits_exp/x_logits_sumexp
                dloss_dx = -log_softmax_jac/nonzero_count
                tl.atomic_add(dloss_dx_row_start_ptr + offsets, dloss_dx, mask=mask)
        
def t_avg_cross_entropy_loss_bkwd3_t(y_labels, x_logits):
    dloss_dx_shape = x_logits.shape 
    y_labels = y_labels.reshape((-1,))
    y_labels = y_labels.to(torch.int64) # TODO XXX: shouldn't we pass y_labels in int64 already?
    x_logits = x_logits.reshape((y_labels.numel(), -1))
    nonzero_count = torch.count_nonzero(y_labels)
    
    n_rows, n_cols = x_logits.shape
    loss = torch.zeros((1), device=x_logits.device) # can we just return value from triton kernel instead? I doubt that
    dloss_dx = torch.zeros_like(x_logits)
    # TODO T: The below numbers were tuned for A10 by choosing num_warps=8
    num_warps=8
    num_stages=2
    BLOCK_SIZE = 1024
    aux_idx = torch.zeros((n_rows, BLOCK_SIZE), device=x_logits.device, dtype=torch.bool)
    aux_idx.scatter_(1, (y_labels % BLOCK_SIZE).unsqueeze(1), True)
    num_programs = min(n_rows, 480)
    
    t_avg_cross_entropy_loss_bkwd3_k[(num_programs,)](y_labels, x_logits, loss, dloss_dx, aux_idx, x_logits.stride(0), dloss_dx.stride(0), aux_idx.stride(0), nonzero_count, 
                                                      n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages)
    
    return loss, nonzero_count, dloss_dx.reshape(dloss_dx_shape)

def accuracy(y_labels, x_logits):
    return torch.nanmean(torch.where(y_labels!=0, y_labels == torch.argmax(x_logits, axis=-1), float('nan')))

# NB: this is WIP, as it hasn't been tested in longer runs yet TODO XXX
def predict(params, y_mask, y_indices, sample_len, start_tok, end_tok):
    y_sample = torch.full((y_mask.shape[0], sample_len), start_tok, device=y_mask.device)
    for step_i in range(sample_len-1):
        new_y = batched_forward_gpt2(params, y_sample, y_mask, y_indices, False)
        new_toks = torch.argmax(new_y[:, step_i], axis=-1)
        y_sample[:,step_i+1]=new_toks
    y_sample = torch.where(torch.cummax(y_sample, axis=1)[0] != end_tok, y_sample, 0) # replace END token, and what follows with padding

    y_sample = y_sample[:, 1:]
    return torch.where(y_sample!=start_tok, y_sample, 0) # It should not be happening, but for random model it might.

def loss(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N
    return _loss(batched_forward_gpt2, avg_cross_entropy_loss, params, y, y_mask, y_indices, train)

def t_loss(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N
    fwd_fn = partial(t_gpt2_forward, p_gen_aux=p_gen_aux)
    return _loss(fwd_fn, t_avg_cross_entropy_loss, params, y, y_mask, y_indices, train)
    
def _loss(fwd_fn, celoss_fn, params, y, y_mask, y_indices, train):  # inputs: BS x N
    y_in = y[:, :-1]
    y_out = y[:, 1:]
    
    # TODO: write it without copying memory? is it possible? 
    logits = fwd_fn(params, y_in, y_mask, y_indices, train)
    loss_val, tokens_count = celoss_fn(y_out, logits)
    acc = accuracy(y_out, logits) # TODO: Do I need to stop_gradient on this? I think not, but double-check
    return loss_val, (loss_val, acc, tokens_count/y_out.numel()) # TODO: this is wrapping, but we could make use of jax.value_and_grad instead

loss_train = partial(loss, train=True)
loss_eval = torch.compile(partial(loss, train=False))

uncompiled_grad_loss = grad(loss_train, has_aux=True) # used for debugging purposes
grad_loss = torch.compile(grad(loss_train, has_aux=True))

def t_loss_bkwd(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N    
    y_in = y[:, :-1]
    y_out = y[:, 1:]
     
    jac_gpt2 = t_gpt2_bkwd_p(params, y_in, y_mask, y_indices, train, p_gen_aux)
    logits = t_gpt2_forward(params, y_in, y_mask, y_indices, train, p_gen_aux) 
    
    jac_celoss = t_avg_cross_entropy_loss_bkwd(y_out, logits)
    loss_val, tokens_count = t_avg_cross_entropy_loss(y_out, logits)
    acc = accuracy(y_out, logits)
    
    dloss_dp = list(jac_gpt2)
    for i in range(len(dloss_dp)):
        dloss_dp[i] = _mult_jacs_in_2d(jac_celoss, dloss_dp[i], logits)
    return dloss_dp, (loss_val, acc, tokens_count/y_out.numel())

def t_loss_bkwd2(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N    
    y_in = y[:, :-1]
    y_out = y[:, 1:]
     
    logits = t_gpt2_forward(params, y_in, y_mask, y_indices, train, p_gen_aux) 
    
    dloss_dx = t_avg_cross_entropy_loss_bkwd2(y_out, logits)
    dloss_dx = t_gpt2_bkwd2_p(dloss_dx, params, y_in, y_mask, y_indices, train, p_gen_aux)
    
    loss_val, tokens_count = t_avg_cross_entropy_loss(y_out, logits)
    acc = accuracy(y_out, logits)
    
    return dloss_dx, (loss_val, acc, tokens_count/y_out.numel())

def t_loss_bkwd3(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N    
    y_in = y[:, :-1]
    y_out = y[:, 1:]
     
    logits, acts = t_gpt2_forward_with_acts(params, y_in, y_mask, y_indices, train, p_gen_aux) 
    
    loss_val, tokens_count, dloss_dx = t_avg_cross_entropy_loss_bkwd3(y_out, logits)
    dloss_dx = t_gpt2_bkwd3_p(dloss_dx, acts, params, y_in, y_mask, y_indices, train, p_gen_aux)
    acc = accuracy(y_out, logits)
    
    return dloss_dx, (loss_val, acc, tokens_count/y_out.numel())

def t_loss_bkwd3_t(params, y, y_mask, y_indices, train, p_gen_aux=None):  # inputs: BS x N    
    y_in = y[:, :-1]
    y_out = y[:, 1:]
     
    logits, acts = t_gpt2_forward_with_acts_t(params, y_in, y_mask, y_indices, train, p_gen_aux) 
    
    loss_val, tokens_count, dloss_dx = t_avg_cross_entropy_loss_bkwd3_t(y_out, logits)
    dloss_dx = t_gpt2_bkwd3_p_t(dloss_dx, acts, params, y_in, y_mask, y_indices, train, p_gen_aux)
    acc = accuracy(y_out, logits)
    
    return dloss_dx, (loss_val, acc, tokens_count/y_out.numel())

# TODO XXX: This is WIP: it hasn't been tested in the longer runs
# TODO XXX: take y_prefix_len which not to ignore probs for?
def log_probs(params, y, y_mask, y_indices):  # inputs: batch_size x seq_len
    y_in = y[:, :-1]
    y_out = y[:, 1:]

    # copied a few lines from avg_cross_entropy_loss # TODO XXX XXX: reuse instead!
    def compute_log_probs(y_labels, x_logits): # y_labels: batch_len x seq_len, x_logits: batch_len x seq_len x vocab_size
        b, n, v = x_logits.shape
        b_n = y_labels.numel()
        y_labels_1d = torch.reshape(y_labels, (-1,)) # there is probably a way of doing it while staying in 2d..
        x_logits_2d = torch.reshape(x_logits, (b_n, -1))
        elements_loss = log_softmax(x_logits_2d)[(torch.arange(b_n), y_labels_1d)]
        elements_loss = torch.where(y_labels_1d != 0, elements_loss, 1) # account for padding tokens
        elements_loss_2d = torch.reshape(elements_loss, (b, n))
        y_log_probs = torch.sum(elements_loss_2d, axis=1)
        return y_log_probs
    
    # TODO: write it without copying memory? is it possible? 
    logits = batched_forward_gpt2(params, y_in, y_mask, y_indices, False) 
    return compute_log_probs(y_out, logits)

# Accumulates gradients in place
#@torch.compile #TODO XXX: What about in_place operations??
def acc_grad_loss(acc_grads, params, y, y_mask, y_indices):
    return _acc_grad_loss(grad_loss, acc_grads, params, y, y_mask, y_indices)

BIG_NR=1_000_000
def sample_p_gen_aux(params):
    nlayers = len(params) - 3
    device = params[0][0].device
    
    n = 1 + 3 * nlayers
    p_gen_aux = torch.randint(0, BIG_NR, (n,), device=device) # TODO XXX: replace with random.random?
    return [it.item() for it in p_gen_aux]

def t_acc_grad_loss(acc_grads, params, y, y_mask, y_indices):
    return _acc_grad_loss(t_loss_bkwd, acc_grads, params, y, y_mask, y_indices)

def t_acc_grad_loss2(acc_grads, params, y, y_mask, y_indices):
    return _acc_grad_loss(t_loss_bkwd2, acc_grads, params, y, y_mask, y_indices)

def t_acc_grad_loss3(acc_grads, params, y, y_mask, y_indices):
    return _acc_grad_loss(t_loss_bkwd3, acc_grads, params, y, y_mask, y_indices)

def t_acc_grad_loss3_t(acc_grads, params, y, y_mask, y_indices):
    return _acc_grad_loss(t_loss_bkwd3_t, acc_grads, params, y, y_mask, y_indices)


def _acc_grad_loss(grad_loss_fn, acc_grads, params, y, y_mask, y_indices):
    i_step_grads, grad_loss_rest = grad_loss_fn(params, y, y_mask, y_indices, train=True, p_gen_aux=sample_p_gen_aux(params))
    
    for grp_i in range(len(acc_grads)):
        for p_i in range(len(acc_grads[grp_i])):
            acc_grads[grp_i][p_i] =  acc_grads[grp_i][p_i] + i_step_grads[grp_i][p_i]
            #acc_grads[grp_i][p_i] =  acc_grads[grp_i][p_i].at[:].add(i_step_grads[grp_i][p_i])
            
    return acc_grads, grad_loss_rest

# ## 
# Optimizers
# ##

# TODO: any call to this function can be replaced by jax's tree_map
def elwise(params_and_grads, func): # generically applying func element-wise
    return [ [ func(*p_and_g) for p_and_g in zip(*p_and_g_items)] for p_and_g_items in zip(*params_and_grads)]

def sgd(params, grads, lr):
    return elwise((params, grads), lambda p,g: p - lr * g)


def init_adam_w(params): # initializes grads and moments estimates
    return elwise((params,), lambda p: torch.zeros_like(p, device="cuda")), [elwise((params,), lambda p: torch.zeros_like(p, device="cuda")) for _ in range(2)]

# @jit
# def adam_w(params, grads, lr, betas, epsilon, moments, i, weight_decay=0.0): #TODO: add implemntation of weight_decay_mask?
#     t = i + 1 # TODO: should we decuple iteration from t, and threading t instead?
#     moments = [elwise((moment, grads), lambda m, g: b*m + (1-b) * pow(g, pow_g)) for b, moment, pow_g in zip(betas, moments, [1,2])]
#     bias_corrected_moments = [elwise((moment,), lambda m: m / (1 - pow(b,t))) for b, moment in zip(betas, moments)]
#     params = elwise((params, *bias_corrected_moments), lambda p,m,v: p - lr *(m / (jnp.sqrt(v) + epsilon) + weight_decay * p))
#     return params, moments

# from functools import partial
# @partial(jax.jit, donate_argnames=("params","moments"), static_argnames=["weight_decay_mask"]) # TODO XXX: ADD torch.compile, how to achieve in-place operations (see below)?
#@torch.compile
def adam_w_in_place(params, grads, lr, betas, epsilon, moments, i, weight_decay=0.0, weight_decay_mask=None):
    t = i + 1 # TODO: should we decuple iteration from t, and threading t instead?

    # TODO: once write it more effiently, combine both loops + vmap (if possible)?
    
    # update moments
    for b, moment, pow_g in zip(betas, moments, [1,2]): 
        for grp_i in range(len(grads)):
            for p_i in range(len(grads[grp_i])):
                #moment[grp_i][p_i] = moment[grp_i][p_i].at[:].multiply(b)
                #moment[grp_i][p_i] = moment[grp_i][p_i].at[:].add((1-b) * pow(grads[grp_i][p_i], pow_g))
                moment[grp_i][p_i] = moment[grp_i][p_i] * b + (1-b) * pow(grads[grp_i][p_i], pow_g)

    # update grads
    for grp_i in range(len(grads)):
        for p_i in range(len(grads[grp_i])):
            bias_correct_func = lambda b, m: m / (1 - pow(b,t))
            m = bias_correct_func(betas[0], moments[0][grp_i][p_i])
            v = bias_correct_func(betas[1], moments[1][grp_i][p_i])
            item_weight_decay = weight_decay if (weight_decay_mask is None or weight_decay_mask[grp_i][p_i]) else 0.0
            #item_weight_decay = weight_decay if weight_decay_mask[grp_i][p_i] else 0.0
            #print(f'Shape {grp_i}, {p_i}: {params[grp_i][p_i].shape} {item_weight_decay}')
            #params[grp_i][p_i] =  params[grp_i][p_i].at[:].add(-lr * (m / (jnp.sqrt(v) + epsilon) + item_weight_decay * params[grp_i][p_i]))
            params[grp_i][p_i] =  params[grp_i][p_i] + (-lr * (m / (torch.sqrt(v) + epsilon) + item_weight_decay * params[grp_i][p_i]))
            
    return params, moments

# # Testing Adam in place:
# # original_pointer = params[0][0].unsafe_buffer_pointer()
# # params, moments = adam_w_in_place(params, grads, lr, betas, epsilon, moments, i)
# # assert params[0][0].unsafe_buffer_pointer() == original_pointer # will not fail
# # params, moments = adam_w(params, grads, lr, betas, epsilon, moments, i)
# # assert params[0][0].unsafe_buffer_pointer() == original_pointer # will fail

# #@jit # TODO: I can't jit that one
def _g_l2norm_squared(g_list):
    return pow(torch.linalg.norm(g_list),2)
def grads_l2norm(grads): # computing l2norm without any memory copies
    return math.sqrt(sum([ sum([_g_l2norm_squared(g) for g in g_items]) for g_items in grads]))
def grads_grps_l2norms(grads):
    return [ math.sqrt(sum([_g_l2norm_squared(g) for g in g_items])) for g_items in grads]

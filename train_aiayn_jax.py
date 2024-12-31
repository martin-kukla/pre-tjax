###############################################
### JAX
###############################################
# UPDATE/TODO XXX: We can now move to jax24.04-py3 (https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-24-04.html)
# TODO: this is slightly faster even with the warning -> invewstigate (current jax version is 0.4.26, where the image has 0.4.17)
#! pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#2024-05-02 08:16:04.763248: W external/xla/xla/service/gpu/nvptx_compiler.cc:718] 
#The NVIDIA driver's CUDA version is 12.2 which is older than the ptxas CUDA version (12.4.131). 
#Because the driver is older than the ptxas version, XLA is disabling parallel compilation, which may slow down compilation. 
#You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.

# TODO: It looks like I am suffering from fragmentation on GPU, thus enabling prelocation
# Disable JAX memory preallocation
import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
#%env XLA_PYTHON_CLIENT_PREALLOCATE=false
#%env XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

#!LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
import jax
jax.devices()


###############################################
### DATASET
###############################################
from tokenized_dataset import load_tokenized_dataset, get_batched_examples, get_batched_examples_packed
ds, (tokenizer, detokenize, tokenizer_vocab_size) = load_tokenized_dataset()
ds = ds.shuffle(seed=42) # TODO: put it in better place? does it mess up with resume_from_checkpoint logic?


###############################################
### MODEL
###############################################
from model_jax import *
import jax.numpy as jnp
from jax import grad, jit, vmap, lax 
from jax import random

LAYERS = 6
model_vocab_size = tokenizer_vocab_size + 3 # add padding token (0) + start of sequence token + end of sequence token 
START_TOK = tokenizer_vocab_size + 1
END_TOK = tokenizer_vocab_size + 2 # TODO: in standard LLM convention, it should be 1. Also, it could be part of tokenizer_vocab_size
EMB_DIM=512
FFN_DIM=2048
NUM_HEADS = 8
params = init_transformer_aiayn(model_vocab_size, EMB_DIM, LAYERS, NUM_HEADS, FFN_DIM, random.PRNGKey(0))

print(f'Vocabulary size: {model_vocab_size}')
num_params = sum([jnp.size(p_leaf) for p_leaf in jax.tree_util.tree_leaves(params)])
print(f'Number of params: {num_params}')


###############################################
### Loss + grads
###############################################
#def one_hot(x, k, dtype=jnp.float32): 
#    """Create a one-hot encoding of x of size k.""" 
#    return jnp.array(x[:, None] == jnp.arange(k), dtype)
#
#batched_one_hot = vmap(one_hot, in_axes=(0, None))

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

from functools import partial

@partial(jax.jit, static_argnames=['sample_len'])
def predict(params, x, x_mask, y_mask, yx_mask, x_indices, y_indices, sample_len): # TODO: code up not-scanned version, which could be faster on GPU
    def predict_step(step_i, y):
        # TODO: Cache key-value pairs
        new_y = batched_forward_aiayn(params, x, y, x_mask, y_mask, yx_mask, x_indices, y_indices, random.PRNGKey(0), False) 
        new_toks = jnp.argmax(new_y[:, step_i], axis=-1)
        y = y.at[:,step_i+1].set(new_toks)
        return y
    
    start_toks = jnp.full((x.shape[0], sample_len), START_TOK)
    y_sample = jax.lax.fori_loop(0, sample_len, predict_step, start_toks) 
    y_sample = jnp.where(jax.lax.cummax(y_sample, axis=1) != END_TOK, y_sample, 0) # replace END token, and what follows with padding

    y_sample = y_sample[:, 1:]
    return jnp.where(y_sample!=START_TOK, y_sample, 0) # It should not be happening, but for random model it might.2

def loss(params, x, y, x_mask, y_mask, yx_mask, x_indices, y_indices, key, train):  # inputs: batch_size x seq_len
    y_in = y[:, :-1]
    y_out = y[:, 1:]
    
    # TODO: write it without copying memory? is it possible? 
    logits = batched_forward_aiayn(params, x, y_in, x_mask, y_mask, yx_mask, x_indices, y_indices, key, train) 
    loss_val, tokens_count = avg_cross_entropy_loss(y_out, logits)
    acc = accuracy(y_out, logits) # TODO: Do I need to stop_gradient on this? I think not, but double-check
    return loss_val, (loss_val, acc, tokens_count/jnp.size(y_out)) # TODO: this is wrapping, but we could make use of jax.value_and_grad instead

loss_train = partial(loss, train=True)
loss_eval = jit(partial(loss, key=random.PRNGKey(0), train=False))

grad_loss = jit(grad(loss_train, has_aux=True))
#grad_loss = grad(loss_train, has_aux=True)

#print(f'iter #{i} loss {loss_train(params, jnp.array(x[:1]), jnp.array(y[:1]), random.PRNGKey(0))[0] }')

#with jax.disable_jit():
#print(f'iter #{i} loss {predict(params, jnp.array(x[:2], 50)) }')


###############################################
### Optimizers
###############################################
# TODO: any call to this function can be replaced by jax's tree_map
def elwise(params_and_grads, func): # generically applying func element-wise
    return [ [ func(*p_and_g) for p_and_g in zip(*p_and_g_items)] for p_and_g_items in zip(*params_and_grads)]

def sgd(params, grads, lr):
    return elwise((params, grads), lambda p,g: p - lr * g)

@jit
def adam(params, grads, lr, betas, epsilon, moments, i):
    t = i + 1 # TODO: should we decuple iteration from t, and threading t instead?
    moments = [elwise((moment, grads), lambda m, g: b*m + (1-b) * pow(g, pow_g)) for b, moment, pow_g in zip(betas, moments, [1,2])]
    bias_corrected_moments = [elwise((moment,), lambda m: m / (1 - pow(b,t))) for b, moment in zip(betas, moments)]
    params = elwise((params, *bias_corrected_moments), lambda p,m,v: p - lr * m / (jnp.sqrt(v) + epsilon))
    return params, moments

from functools import partial
@partial(jax.jit, donate_argnames=("params","moments"))
def adam_in_place(params, grads, lr, betas, epsilon, moments, i):
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
            params[grp_i][p_i] =  params[grp_i][p_i].at[:].add(-lr * m / (jnp.sqrt(v) + epsilon))
            
    return params, moments

# Testing Adam in place:
#original_pointer = params[0][0].unsafe_buffer_pointer()
#params, moments = adam_in_place(params, grads, lr, betas, epsilon, moments, i)
#assert params[0][0].unsafe_buffer_pointer() == original_pointer # will not fail
#params, moments = adam(params, grads, lr, betas, epsilon, moments, i)
#assert params[0][0].unsafe_buffer_pointer() == original_pointer # will fail


###############################################
### Infra utils
###############################################
def print_mem_stats():
    mem_stats = jax.devices()[0].memory_stats()
    conv = lambda k: mem_stats[k] / pow(1000,3)
    print(f'GB in use: {conv("bytes_in_use")}. GB limit: {conv("bytes_limit")}')

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="t",

    # track hyperparameters and run metadata
    #config={
    #"learning_rate": 0.02,
    #"architecture": "CNN",
    #"dataset": "CIFAR-100",
    #"epochs": 10,
    #}
    sync_tensorboard=True
)


###############################################
### Training loop
###############################################
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import itertools
import pickle
import evaluate

# Infra training params
run_name = datetime.datetime.now().strftime("%h%d_%H-%M-%S")
# Since implementation of gradient accumulation is very primitve, we need logging & checkpoint steps params
# to be multiplication of gradient_accumulations_steps. 
# TODO: Introduce effective step (conditioned on accumulation steps), and do logging/checkpoint in respect to effective  step
log_every_steps = 16
eval_every_steps = 4000 #500 * 8 machines
eval_n_examples = 4
writer = SummaryWriter(f'/lego/storage/output/runs/{run_name}')
#checkpoint_every_steps = None #500 * 8 machines
checkpoint_every_steps = 4000 #20000
resume_from_checkpoint = None
#resume_from_checkpoint = 'runs/Jun07_10-12-10/checkpoint_4000.pkl' # TODO: Confirm runs from checkpoints are still fully reproducible


# ML training params
key_training = random.PRNGKey(0) 
batch_size= 512 #416 # TODO: Investigate OOMs when 496? #512
gradient_accumulations_steps = 8 # to imitate paper's 8 devices
num_steps = 800000 # paper's 100k steps *  8 devices
lr = 0.001 # Effectively ignored if lr scheduler is used (i.e. warmup_steps is set to something else than None)
warmup_steps= 4000
betas = (0.9, 0.98) 
epsilon = 10e-9
moments = [elwise((params,), lambda p: jnp.zeros_like(p)) for _ in range(2)] # moment esimtates
seq_len = 50 # TODO: 124 is maximum length in validation dataset. 
x_tokens_per_batch = 15000 #For variable batch len, we don't use it as we can fit less data (paper does 25k)

x_eval, y_eval, x_eval_mask, y_eval_mask, yx_eval_mask, x_eval_indices, y_eval_indices  = next(get_batched_examples(ds, eval_n_examples, seq_len, START_TOK, END_TOK, "validation")) 

i = 0 
ds_train_rows_read = 0
if resume_from_checkpoint is not None:
    with open(resume_from_checkpoint,'rb') as f:
        i, ds_train_rows_read, params, moments, key_training = pickle.load(f)   
        print(f'Resuming training from the checkpoint: i {i} ds_train_rows_read {ds_train_rows_read}')

num_params = sum([jnp.size(p_leaf) for p_leaf in jax.tree_util.tree_leaves(params)])
print(f'Number of params: {num_params}')
grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

from functools import partial
@partial(jax.jit, donate_argnames=("acc_grads"))
def acc_grad_loss(acc_grads, params, x, y, x_mask, y_mask, yx_mask, x_indices, y_indices, key_iter):
    i_step_grads, grad_loss_rest = grad_loss(params, x, y, x_mask, y_mask, yx_mask, x_indices, y_indices, key_iter)
    
    for grp_i in range(len(acc_grads)):
        for p_i in range(len(acc_grads[grp_i])):
            acc_grads[grp_i][p_i] =  acc_grads[grp_i][p_i].at[:].add(i_step_grads[grp_i][p_i])
            
    return acc_grads, grad_loss_rest

while True:
    #for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0):
    for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples_packed(ds, batch_size, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0):
        x, y, x_mask, y_mask, yx_mask, x_indices, y_indices = batch
        # Training step
        # TODO: introduce update func, which does grad_loss and adam, and then call/jit that function instead of calling/jitting two separate ones
        key_training, key_iter = random.split(key_training, 2)
        grads, (loss_val, acc, _) = acc_grad_loss(grads, params, jnp.array(x), jnp.array(y), jnp.array(x_mask), jnp.array(y_mask), jnp.array(yx_mask), jnp.array(x_indices), jnp.array(y_indices), key_iter)
        #grads, (loss_val, acc) = grad_loss(params, jnp.array(x), jnp.array(y), key_iter)
        
        #params = sgd(params, grads, lr)
        if warmup_steps is not None:
            i_multidevice = i // gradient_accumulations_steps
            lr = pow(EMB_DIM, -0.5) * min(pow((i_multidevice+1), -0.5), (i_multidevice+1) * pow(warmup_steps, -1.5))

        if i > 0 and i % gradient_accumulations_steps == 0:
            for grp_i in range(len(grads)):
                for p_i in range(len(grads[grp_i])):
                    grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].divide(gradient_accumulations_steps)
            
            #params, moments = adam(params, grads, lr, betas, epsilon, moments, i)
            params, moments = adam_in_place(params, grads, lr, betas, epsilon, moments, i)
    
        # Logging:
        if i%log_every_steps==0:
            loss_val = loss_val.item()
            acc = acc.item()
            
            import math
            #@jit # TODO: I can't jit that one
            def g_l2norm_squared(g_list):
                return pow(jnp.linalg.norm(g_list),2)
            def l2norm(grads): # computing l2norm without any memory copies
                return math.sqrt(sum([ sum([g_l2norm_squared(g) for g in g_items]) for g_items in grads]))
            def grps_l2norms(grads):
                return [ math.sqrt(sum([g_l2norm_squared(g) for g in g_items])) for g_items in grads]
            grad_norm = l2norm(grads)
            grps_grad_norms = grps_l2norms(grads)

            
            #print(f'iter #{i} loss {loss_val} acc {acc} lr {lr} grad_norm {grad_norm}')
            #print_mem_stats() # TODO: monitor it in tensorboard?
            writer.add_scalar('train/loss', loss_val, i)
            writer.add_scalar('train/acc', acc, i)
            writer.add_scalar('train/lr', lr, i)
            writer.add_scalar('train/grad_norm', grad_norm, i)
            for grp_i, grp_grad_norm in enumerate(grps_grad_norms):
                writer.add_scalar(f'train_details/grad_norm_grp_{grp_i}', grp_grad_norm, i)

            # TODO: some metrics computed on x, other on y. Make it consistent
            #pad_tokens_prop = sum([y_row.count(0) for y_row in y]) / sum([len(y_row) for y_row in y])
            import numpy as np
            pad_tokens_prop = np.count_nonzero(y==0) / y.size
            writer.add_scalar('train_data/pad_tokens_prop', pad_tokens_prop, i)
            writer.add_scalar('train_data/batch_size', len(x), i)
            writer.add_scalar('train_data/batch_seq_len', len(x[0]), i)
            writer.add_scalar('train_data/batch_total_tokens', len(x) * len(x[0]), i)

        # Zeroed accumulated grads: we have to do it after computing grad norms
        if i > 0 and i % gradient_accumulations_steps == 0: 
            for grp_i in range(len(grads)):
                for p_i in range(len(grads[grp_i])):
                    grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].set(0)
            
        # Evaluation
        if i>0 and i%eval_every_steps==0:
            val_losses = []
            val_accs = []
            val_toks_props = []
            for eval_step, batch in enumerate(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, split="validation")): 
                x, y, x_mask, y_mask, yx_mask, x_indices, y_indices = batch
                _, (loss_val, acc, toks_prop) = loss_eval(params, jnp.array(x), jnp.array(y), jnp.array(x_mask), jnp.array(y_mask), jnp.array(yx_mask), jnp.array(x_indices), jnp.array(y_indices))
                val_losses.append(loss_val)
                val_accs.append(acc)
                val_toks_props.append(toks_prop)
            writer.add_scalar('eval/loss', jnp.average(jnp.hstack(val_losses), weights = jnp.hstack(val_toks_props)).item(), i)
            writer.add_scalar('eval/acc', jnp.average(jnp.hstack(val_accs), weights = jnp.hstack(val_toks_props)).item(), i)
            
            # Few predictions
            y_sample = predict(params, jnp.array(x_eval), jnp.array(x_eval_mask), jnp.array(y_eval_mask), jnp.array(yx_eval_mask), jnp.array(x_eval_indices), jnp.array(y_eval_indices), seq_len)
            y_sample = tuple([item.tolist() for item in y_sample])
            def detokenize_y_in(y):
                y_out = y[:, 1:]
                y_out[y_out == END_TOK] = 0
                return detokenize(y_out)
            for detokenized_x_eval, detokenized_y_eval, detokenized_y_sample in zip(detokenize(x_eval), detokenize_y_in(y_eval), detokenize(y_sample)):
                print(f'X:{detokenized_x_eval}\tY: {detokenized_y_eval} \tPREDS: {detokenized_y_sample}\n')

            # Compute BLEU score (it takes around around half a minute)
            references = [] 
            predictions = []
            for _, batch in enumerate(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, split="validation")): 
                x, y, x_mask, y_mask, yx_mask, x_indices, y_indices = batch
                y_sample = predict(params, jnp.array(x), jnp.array(x_mask), jnp.array(y_mask), jnp.array(yx_mask), jnp.array(x_indices), jnp.array(y_indices), seq_len)
                y_sample = tuple([item.tolist() for item in y_sample])

                for detokenized_x, detokenized_y, detokenized_y_sample in zip(detokenize(x), detokenize_y_in(y), detokenize(y_sample)):
                    references.append(detokenized_y)
                    predictions.append(detokenized_y_sample)
            
            bleu = evaluate.load("bleu")
            try: # HotFix: For small eval_steps, it's possible to get ZeroDivisionError
                results = bleu.compute(predictions=predictions, references=references)
            except ZeroDivisionError:
                results = {"bleu": -1, "precisions": [-1, -1, -1, 1]}
            writer.add_scalar('eval/bleu', results['bleu'], i)
            writer.add_scalar('eval/bleu.precision0', results['precisions'][0], i)
            writer.add_scalar('eval/bleu.precision1', results['precisions'][1], i)
            writer.add_scalar('eval/bleu.precision2', results['precisions'][2], i)
            writer.add_scalar('eval/bleu.precision3', results['precisions'][3], i)
                
        i = i + 1
        ds_train_rows_read = ds_train_rows_read + len(x)

        # Checkpointing (i, ds_train_rows_read, params, moments).
        if checkpoint_every_steps is not None and (i>0 and i%checkpoint_every_steps==0):
            import os
            training_state = (i, ds_train_rows_read, params, moments, key_training)
            filename = f'runs/{run_name}/checkpoint_{i}.pkl'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(training_state, f)

        if i> num_steps:
            break
                
    ds_train_rows_read=0 # After each epoch, reset dataset pointer

writer.close()


###############################################
### Final test predictions + BLEU computation
###############################################
print(f'Few predictions for validation dataset')
y_sample = predict(params, jnp.array(x_eval))
y_sample = tuple([item.tolist() for item in y_sample])
for detekonized_x_eval, detokenized_y_eval, detokenized_y_sample in zip(detokenize(x_eval), detokenize(y_eval), detokenize(y_sample)):
    print(f'X:{detekonized_x_eval}\tY: {detokenized_y_eval} \tPREDS: {detokenized_y_sample}\n')
    references.append(detokenized_y_eval)
    predictions.append(detokenized_y_sample)

print(f'Computing BLEU for validation dataset')
import evaluate
references = [] 
predictions = []
for _, (x, y) in tqdm(enumerate(get_batched_examples_per_length(ds, x_tokens_per_batch, split="validation"))):
    y_sample = predict(params, jnp.array(x), seq_len)
    y_sample = tuple([item.tolist() for item in y_sample])
    for detekonized_x_eval, detokenized_y_eval, detokenized_y_sample in zip(detokenize(x), detokenize(y), detokenize(y_sample)):
        references.append(detokenized_y_eval)
        predictions.append(detokenized_y_sample)

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)
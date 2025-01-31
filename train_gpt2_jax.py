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
### DATASETs
###############################################
import datasets
from tokenized_dataset import load_tokenized_dataset_gpt2, load_tokenized_dataset_hellaswag, unpack_hellaswag_x, unpack_hellaswag_batched_x, concatenate_hellaswag_y_and_choice, get_batched_examples, get_batched_examples_packed 
ds, (tokenize, detokenize, tokenizer_vocab_size) = load_tokenized_dataset_gpt2("train[:10%]") #:1% or :1000
ds = ds.train_test_split(test_size=0.01, seed=42) # TODO: put seed in better place? does it mess up with resume_from_checkpoint logic?
ds = datasets.DatasetDict({
    'train': ds['train'],
    'validation': ds['test'] #rename
})
print(ds)

# Some stats on HellaSwag. Given the tokenicer: 
# Max len of concatenated y+longest choice is 149
# Max sum of choices tokens lens is 263 (Important for flattening choices in x + seq_len param for data collactor)
hellaswag_ds = load_tokenized_dataset_hellaswag(tokenize)
print(hellaswag_ds)

# Tests:
# item = next(x for x in hellaswag_ds)
# print(item)
# print(detokenize((item['y'],)))
# item_x = item['x']
# choices, label = unpack_hellaswag_x(item['x'])
# print(detokenize(choices)) # TODO XXX: one of chocies has ", while others have '. Is it anything serious?
# print(label)


###############################################
### Model
###############################################
from model_jax import *
import jax.numpy as jnp
from jax import random

LAYERS = 12
model_vocab_size = tokenizer_vocab_size + 3 # add padding token (0) + start of sequence token + end of sequence token 
START_TOK = tokenizer_vocab_size + 1
END_TOK = tokenizer_vocab_size + 2 # TODO: in default LLM convention, it should be 1. Also, it could be part of tokenizer_vocab_size
EMB_DIM=768
FFN_DIM=3072
NUM_HEADS = 12
seq_len= 512 # TODO XXX: 1024 is orginal paper
params = init_transformer_gpt2(model_vocab_size, EMB_DIM, LAYERS, NUM_HEADS, FFN_DIM, seq_len, random.PRNGKey(0))

print(f'Vocabulary size: {model_vocab_size:_}')
print(f'Number of params: {count_num_params(params):_}')

### Loss + Grads + Optimizers
from loss_and_optimizer_jax import loss_train, loss_eval, log_probs, grad_loss, predict, acc_grad_loss, init_adam_w, adam_w_in_place, grads_l2norm, grads_grps_l2norms

# Figure out non bias/gain params, as we only want to apply weight decay to those in AdamW
# Only 1D weights, which are initialized to 0s are bias/gain params (including bias of LayerNorm)
weight_decay_mask = tuple([ tuple([not (item.ndim==1 and all(item==0)) for item in grp]) for grp in params])
print(weight_decay_mask)


###############################################
### Infra utils
###############################################
def print_mem_stats():
    mem_stats = jax.devices()[0].memory_stats()
    conv = lambda k: mem_stats[k] / pow(1000,3)
    print(f'GB in use: {conv("bytes_in_use")}. GB limit: {conv("bytes_limit")}')

import wandb

# start a new wandb run to track this script
if True:
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
import numpy as np # should we get rid of it?
import math

# Infra training params
run_name = datetime.datetime.now().strftime("%h%d_%H-%M-%S")
log_every_steps_multidevice = 10
eval_every_steps_multidevice = 500
eval_n_examples = 4
writer = SummaryWriter(f'/lego/storage/output/runs/{run_name}')
#checkpoint_every_steps = None #500 * 8 machines
checkpoint_every_steps = 4000 #20000 # TODO: move to use _multidevice too
resume_from_checkpoint = None
#resume_from_checkpoint = 'runs/Jun07_10-12-10/checkpoint_4000.pkl' # TODO: Confirm runs from checkpoints are still fully reproducible


# ML training params
key_training = random.PRNGKey(0) 
batch_size= 16 
gradient_accumulations_steps = 16 # TODO XXX: This means effective batch_size=256 instead of 512 used in the paper
num_steps_multidevice = 50000 #30000 #10000 #100000 # TODO XXX: think what it should be for GPT2
max_lr = 0.00025
warmup_steps_multidevice= 2000
betas = (0.9, 0.98) 
epsilon = 10e-9
grads, moments = init_adam_w(params)

# TODO XXX: remove below one
_, _, _, y_eval_mask, _, _, y_eval_indices  = next(get_batched_examples(ds, eval_n_examples, seq_len, START_TOK, END_TOK, "validation")) 
    
i = 0 
ds_train_rows_read = 0
if resume_from_checkpoint is not None:
    with open(resume_from_checkpoint,'rb') as f:
        i, ds_train_rows_read, params, moments, key_training = pickle.load(f)   
        print(f'Resuming training from the checkpoint: i {i} ds_train_rows_read {ds_train_rows_read}')

print(f'Number of params: {count_num_params(params):_}')

num_steps = num_steps_multidevice * gradient_accumulations_steps
while True:
    #for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0):
    for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples_packed(ds, batch_size, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0.3):
        _, y, _, y_mask, _, _, y_indices = batch
        # Training step
        # TODO: introduce update func, which does grad_loss and adam, and then call/jit that function instead of calling/jitting two separate ones
        key_training, key_iter = random.split(key_training, 2)
        grads, (loss_val, acc, _) = acc_grad_loss(grads, params, jnp.array(y), jnp.array(y_mask), jnp.array(y_indices), key_iter)
        #grads, (loss_val, acc) = grad_loss(params, jnp.array(x), jnp.array(y), key_iter)

        # LR Scheduler
        #lr = max_lr # for SGD

        i_multidevice = i // gradient_accumulations_steps
        is_i_device_zero = i % gradient_accumulations_steps == 0

        # AIAYN:
        #lr = pow(EMB_DIM, -0.5) * min(pow((i_multidevice+1), -0.5), (i_multidevice+1) * pow(warmup_steps, -1.5))

        # GPT1:
        if i_multidevice < warmup_steps_multidevice:
            lr = (i_multidevice+1)/warmup_steps_multidevice * max_lr
        else:
            t_step = i_multidevice - warmup_steps_multidevice
            t_max = num_steps_multidevice - warmup_steps_multidevice
            lr = max_lr * (1 + math.cos(math.pi * t_step/t_max))/2

        #params = sgd(params, grads, lr)
        if i > 0 and i % gradient_accumulations_steps == 0:
            for grp_i in range(len(grads)):
                for p_i in range(len(grads[grp_i])):
                    grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].divide(gradient_accumulations_steps)
            
            #params, moments = adam_w(params, grads, lr, betas, epsilon, moments, i)
            params, moments = adam_w_in_place(params, grads, lr, betas, epsilon, moments, i, weight_decay=0.01, weight_decay_mask=weight_decay_mask)
    
        # Logging:
        if i_multidevice%log_every_steps_multidevice==0 and is_i_device_zero:
            loss_val = loss_val.item()
            acc = acc.item()
            
            grad_norm = grads_l2norm(grads)
            grps_grad_norms = grads_grps_l2norms(grads)

            
            #print(f'iter #{i} loss {loss_val} acc {acc} lr {lr} grad_norm {grad_norm}')
            #print_mem_stats() # TODO: monitor it in tensorboard?
            writer.add_scalar('train/loss', loss_val, i_multidevice)
            writer.add_scalar('train/acc', acc, i_multidevice)
            writer.add_scalar('train/lr', lr, i_multidevice)
            writer.add_scalar('train/grad_norm', grad_norm, i_multidevice)
            for grp_i, grp_grad_norm in enumerate(grps_grad_norms):
                writer.add_scalar(f'train_details/grad_norm_grp_{grp_i}', grp_grad_norm, i_multidevice)

            # TODO: some metrics computed on x, other on y. Make it consistent
            #pad_tokens_prop = sum([y_row.count(0) for y_row in y]) / sum([len(y_row) for y_row in y])
            pad_tokens_prop = np.count_nonzero(y==0) / y.size
            writer.add_scalar('train_data/pad_tokens_prop', pad_tokens_prop, i_multidevice)
            writer.add_scalar('train_data/batch_size', len(y), i_multidevice)
            writer.add_scalar('train_data/batch_seq_len', len(y[0]), i_multidevice)
            writer.add_scalar('train_data/batch_total_tokens', len(y) * len(y[0]), i_multidevice)

        # Zeroed accumulated grads: we have to do it after computing grad norms
        if i > 0 and i % gradient_accumulations_steps == 0: 
            for grp_i in range(len(grads)):
                for p_i in range(len(grads[grp_i])):
                    grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].set(0)
            
        # Evaluation
        if i_multidevice>0 and i_multidevice%eval_every_steps_multidevice==0 and is_i_device_zero:
            val_losses = []
            val_accs = []
            val_toks_props = []
            for eval_step, batch in enumerate(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, split="validation")): 
                _, y, _, y_mask, _, _, y_indices = batch
                _, (loss_val, acc, toks_prop) = loss_eval(params, jnp.array(y), jnp.array(y_mask), jnp.array(y_indices))
                val_losses.append(loss_val)
                val_accs.append(acc)
                val_toks_props.append(toks_prop)
            writer.add_scalar('eval/loss', jnp.average(jnp.hstack(val_losses), weights = jnp.hstack(val_toks_props)).item(), i_multidevice)
            writer.add_scalar('eval/acc', jnp.average(jnp.hstack(val_accs), weights = jnp.hstack(val_toks_props)).item(), i_multidevice)
            
            # Few predictions TODO XXX: vary temperature -> diff samples
            y_sample = predict(params, jnp.array(y_eval_mask), jnp.array(y_eval_indices), seq_len, START_TOK, END_TOK)
            y_sample = tuple([item.tolist() for item in y_sample])
            def detokenize_y_in(y):
                y_out = y[:, 1:]
                y_out[y_out == END_TOK] = 0
                return detokenize(y_out)
            for detokenized_y_sample in detokenize(y_sample):
                print(f'PREDS: {detokenized_y_sample}\n')

            # Compute HellaSwag score
            print(f'Compute HellaSwag score')
            hellaswag_accs = [] # TODO XXX: enable seq_len be different for x vs y; 
            num_hellaswag_batches = 100 #TODO XXX:; run for the whole dataset
            for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples(hellaswag_ds, batch_size, seq_len, START_TOK, END_TOK, split=None), num_hellaswag_batches))):
                choices_vals = []
                x, y, _, y_mask, _, _, y_indices = batch
                choices, labels = unpack_hellaswag_batched_x(x) 
                
                for choice in choices:
                    y, y_mask = concatenate_hellaswag_y_and_choice(y, choice, END_TOK) # no need to return new y_indices for now.
                    choice_log_probs = log_probs(params, jnp.array(y), jnp.array(y_mask), jnp.array(y_indices))
                    choices_vals.append(choice_log_probs)
                choices_vals = np.array(choices_vals).transpose() # we want choice per column
                hellaswag_accs.extend(np.argmax(choices_vals, axis=1)==labels)
                   
            hellaswag_acc = sum(hellaswag_accs)/len(hellaswag_accs)
            print(f'HellaSwag score:', hellaswag_acc)
            writer.add_scalar('eval/hellaswag', hellaswag_acc, i_multidevice)
                
        i = i + 1
        ds_train_rows_read = ds_train_rows_read + len(y)

        # Checkpointing (i, ds_train_rows_read, params, moments).
        # TODO XXX: I haven't used it for a while, and likely it's not working.. probably we can delete 
        if checkpoint_every_steps is not None and (i>0 and i%checkpoint_every_steps==0):
            import os
            training_state = (i, ds_train_rows_read, params, moments, key_training)
            filename = f'runs/{run_name}/checkpoint_{i}.pkl'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(training_state, f)
                
    ds_train_rows_read=0 # After each epoch, reset dataset pointer

writer.close()


###############################################
### FOR TESTING
###############################################

# Compute HellaSwag score
import numpy as np

hellaswag_accs = []
from tqdm import tqdm
import itertools
for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples(hellaswag_ds, 2, seq_len, START_TOK, END_TOK, split=None), 4))): 
#for _, batch in tqdm(enumerate(get_batched_examples(hellaswag_ds, 1, 400, START_TOK, END_TOK, split=None))):
    choices_vals = []
    x, y, _, y_mask, _, _, y_indices = batch
    choices, labels = unpack_hellaswag_batched_x(x)
    
    for choice in choices:
        y, y_mask = concatenate_hellaswag_y_and_choice(y, choice, END_TOK) # no need to return new y_indices for now.
        choice_log_probs = log_probs(params, jnp.array(y), jnp.array(y_mask), jnp.array(y_indices))
        choices_vals.append(choice_log_probs)
    choices_vals = np.array(choices_vals).transpose()
    hellaswag_accs.extend(np.argmax(choices_vals, axis=1)==labels)

#print("hellaswag_accs", hellaswag_accs)
hellaswag_acc = sum(hellaswag_accs)/len(hellaswag_accs)
print(hellaswag_acc)


###############################################
### Final test predictions + BLEU computation
###############################################
x_tokens_per_batch = 15000 #For variable batch len, we don't use it as we can fit less data (paper does 25k)

print(f'Few predictions for validation dataset')
y_sample = predict(params, jnp.array(x_eval), seq_len, START_TOK, END_TOK)
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
    y_sample = predict(params, jnp.array(x), seq_len, START_TOK, END_TOK)
    y_sample = tuple([item.tolist() for item in y_sample])
    for detekonized_x_eval, detokenized_y_eval, detokenized_y_sample in zip(detokenize(x), detokenize(y), detokenize(y_sample)):
        references.append(detokenized_y_eval)
        predictions.append(detokenized_y_sample)

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)
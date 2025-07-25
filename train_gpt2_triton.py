###############################################
# This script can be used for training GPT2 model using
# two different ml engines: torch.func+jit and triton.
# See below for specifying correct ml engine
###############################################
import argparse
parser = argparse.ArgumentParser("train_gpt2_trition")
parser.add_argument("backend", help="Either 'torchfunc_jit', 'triton', 'pre-triton' or 'debug_jacs'.", type=str)
parser.add_argument("--test", action='store_true')
parser.add_argument("--profile", action='store_true')
parser.add_argument("--eval_only", action='store_true')
parser.add_argument("--from_checkpoint", type=str, default=None)
args = parser.parse_args()
assert args.backend in ["torchfunc_jit", "triton", "pre-triton", "debug_jacs"]

###############################################
### DATASETs
###############################################
import datasets
from tokenized_dataset import load_tokenized_dataset_gpt2, load_tokenized_dataset_hellaswag, unpack_hellaswag_x, unpack_hellaswag_batched_x, concatenate_hellaswag_y_and_choice, get_batched_examples, get_batched_examples_packed 
ds, (tokenize, detokenize, tokenizer_vocab_size) = load_tokenized_dataset_gpt2("train[:5%]") #:1% or :1000
ds = ds.train_test_split(test_size=0.02, seed=42) # TODO: put seed in better place? does it mess up with resume_from_checkpoint logic?
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


###############################################
### Model
###############################################
import torch
from model_triton import init_transformer_gpt2, count_num_params

LAYERS = 12
model_vocab_size = tokenizer_vocab_size + 3 # add padding token (0) + start of sequence token + end of sequence token 
START_TOK = tokenizer_vocab_size + 1
END_TOK = tokenizer_vocab_size + 2 # TODO: in default LLM convention, it should be 1. Also, it could be part of tokenizer_vocab_size
EMB_DIM=768
FFN_DIM=EMB_DIM * 4
NUM_HEADS = 12
seq_len= 512 # TODO XXX: 1024 is orginal paper
params = init_transformer_gpt2(model_vocab_size, EMB_DIM, LAYERS, NUM_HEADS, FFN_DIM, seq_len)

print(f'Vocabulary size: {model_vocab_size:_}')
print(f'Number of params: {count_num_params(params):_}')

# ### Loss + Grads + Optimizers
from loss_and_optimizer_triton import loss_train, loss_eval, log_probs, grad_loss, predict, acc_grad_loss, t_acc_grad_loss, t_acc_grad_loss2, t_acc_grad_loss3, t_acc_grad_loss3_t, init_adam_w, adam_w_in_place, grads_l2norm, grads_grps_l2norms

# Choose the accumulation gradient loss function depending on the selected ML backend
# TODO XXX: differentiate forward func too
if args.backend =="torchfunc_jit":
    acc_grad_loss_func =  acc_grad_loss
elif args.backend =="debug_jacs":
    acc_grad_loss_func = t_acc_grad_loss
elif args.backend =="pre-triton":
    acc_grad_loss_func = t_acc_grad_loss3
else:
    acc_grad_loss_func = t_acc_grad_loss3_t

# # Figure out non bias/gain params, as we only want to apply weight decay to those in AdamW
# # Only 1D weights, which are initialized to 0s are bias/gain params (including bias of LayerNorm)
weight_decay_mask = tuple([ tuple([not (item.ndim==1 and all(item==0)) for item in grp]) for grp in params])
#print(weight_decay_mask)


###############################################
### Tests (forward and backward pass + memory)
###############################################
if args.test:
    # Testing forward pass for triton (eval only)
    # Remaining TODOs: 1) support packed batches in Triton's FlashAttention. Update 7/2: Is this still remaining issue? see comment below..
    from model_torch_func import batched_forward_gpt2
    from model_triton import t_gpt2_forward_with_acts_t
    test_batch_size = 4
    # Note, both version pass the test. I am still wondering whether _packed is properly supported in torch/tirton world
    # JAX version learns slightly faster, which could be explainable by its capability to handle _packed unlike torch/triton. 
    # At this point, it's just hypothesis
    #_, y, _, y_mask, _, _, y_indices = next(get_batched_examples_packed(ds, test_batch_size, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = 0))
    _, y, _, y_mask, _, _, y_indices = next(get_batched_examples(ds, test_batch_size, seq_len, START_TOK, END_TOK, skip_n_rows = 0))

    y = torch.tensor(y, dtype=torch.int32, device="cuda")
    y_mask = torch.tensor(y_mask, dtype=torch.bool, device="cuda")
    y_indices = torch.tensor(y_indices, dtype=torch.int16, device="cuda")

    ## FORWARD test
    y_in = y[:, :-1]
    y_out = y[:, 1:]
    logits_torch_func = batched_forward_gpt2(params, y_in, y_mask, y_indices, False) 
    logits_triton, _ = t_gpt2_forward_with_acts_t(params, y_in, y_mask, y_indices, False) 
    # compare only on non-padded positions:
    logits_torch_func=torch.where(y_out.unsqueeze(2)!=0, logits_torch_func, 0)
    logits_triton=torch.where(y_out.unsqueeze(2)!=0, logits_triton, 0)  

    assert torch.allclose(logits_torch_func, logits_triton, rtol=1e-2, atol=5e-3), (logits_torch_func.shape, logits_triton.shape, logits_torch_func[-2:, -4:, -10:], logits_triton[-2:, -4:, -10:])
    print ("Forward pass test successful")


    ## BACKWARD test
    from loss_and_optimizer_triton import t_loss_bkwd3_t, uncompiled_grad_loss, sample_p_gen_aux
    from loss_and_optimizer_triton import _g_l2norm_squared
    import math
    def grad_l2norms(grads_grps):
        grad_l2norms = []
        for i, g_grp in enumerate(grads_grps):
            i_grad_l2norms = []
            shapes = []
            for j, g in enumerate(g_grp):
                grad_l2norm = math.sqrt(_g_l2norm_squared(g)) #math.sqrt(sum(_g_l2norm_squared(g)))
                i_grad_l2norms.append(grad_l2norm)
                shapes.append(g.shape)
            #print('i, i_grad_l2norms', i, i_grad_l2norms)
            #print('i, shapes', i, shapes)
            grad_l2norms.append(i_grad_l2norms)
        return grad_l2norms

    # Compare loss values
    grads_torch_func, (loss_val_torch_func, acc_torch_func, _) =  uncompiled_grad_loss(params, y, y_mask, y_indices, train=True, p_gen_aux = sample_p_gen_aux(params))
    grads_triton, (loss_val_triton, acc_triton, _) = t_loss_bkwd3_t(params, y, y_mask, y_indices, train=True, p_gen_aux = sample_p_gen_aux(params))
    print(f'Loss torch_func: {loss_val_torch_func}, triton: {loss_val_triton}')
    rtol, atol = 5e-3, 1e-4 #5e-4, 1e-5 # Because of the dropout, we need to relax it (if dropout set to 0, use the latter)
    assert torch.allclose(loss_val_torch_func, loss_val_triton, rtol=rtol, atol=atol), (loss_val_torch_func, loss_val_triton) 
    
    # Compare grad l2 norms
    grad_l2norms_torch_func = grad_l2norms(grads_torch_func)
    grad_l2norms_triton = grad_l2norms(grads_triton)
    for a, b in zip(grad_l2norms_torch_func, grad_l2norms_triton):
        torch.allclose(torch.tensor(a), torch.tensor(b))
    print('Backward pass test successful')


    # # Testing Memory Usage. I decided not to get too deep into it, since this uses torch.grad + torch.compile..
    # TODO XXX: It's still puzzling that the maximum batch_size which torchfunc's version can do is 8 in comparison to 16 by JAX...
    # import math

    # def convert_size(size_bytes):
    #    if size_bytes == 0:
    #        return "0B"
    #    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    #    i = int(math.floor(math.log(size_bytes, 1024)))
    #    p = math.pow(1024, i)
    #    s = round(size_bytes / p, 2)
    #    return "%s %s" % (s, size_name[i])

    # def print_mem():
    #     print(convert_size(torch.cuda.memory_allocated()))

    # print_mem()
    # _, y, _, y_mask, _, _, y_indices = next(get_batched_examples_packed(ds, 8, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = 0))
    # y = torch.tensor(y, dtype=torch.int32, device="cuda")
    # y_mask = torch.tensor(y_mask, dtype=torch.bool, device="cuda")
    # y_indices = torch.tensor(y_indices, dtype=torch.int16, device="cuda")
    # print_mem()
    # loss_val, (_, acc, _) = loss_train(params, y, y_mask, y_indices)
    # #grads, (loss_val, acc, _) = grad_loss(params, y, y_mask, y_indices)
    # print_mem()
    
    # ugly..
    import sys
    sys.exit(0)
    
###############################################
### Profile (forward pass)
###############################################
if args.profile:
    from model_triton import t_gpt2_forward_with_acts_t
    test_batch_size = 8
    #_, y, _, y_mask, _, _, y_indices = next(get_batched_examples_packed(ds, test_batch_size, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = 0))
    _, y, _, y_mask, _, _, y_indices = next(get_batched_examples(ds, test_batch_size, seq_len, START_TOK, END_TOK, skip_n_rows = 0))

    y = torch.tensor(y, dtype=torch.int32, device="cuda")
    y_mask = torch.tensor(y_mask, dtype=torch.bool, device="cuda")
    y_indices = torch.tensor(y_indices, dtype=torch.int16, device="cuda")

    ## FORWARD test
    y_in = y[:, :-1]
    y_out = y[:, 1:]
    
    N=100
    from torch.profiler import profile, record_function, ProfilerActivity
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(N):
            logits_triton, _ = t_gpt2_forward_with_acts_t(params, y_in, y_mask, y_indices, False) 
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20, max_name_column_width=125, top_level_events_only=True, header="Order by CPU"))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=125, top_level_events_only=True, header="Order by GPU"))

    ## BACKWARD test
    
    # Sanity check that backward pass dominates training loop. Indeed, this is a case around 3.25 it/s
    #from tqdm import tqdm
    #for _ in tqdm(range(N)):
    #    grads_triton, (loss_val_triton, acc_triton, _) = t_loss_bkwd3_t(params, y, y_mask, y_indices, train=True, p_gen_aux = sample_p_gen_aux(params))
        
    #from loss_and_optimizer_triton import t_loss_bkwd3_t, sample_p_gen_aux
    #grads_triton, (loss_val_triton, acc_triton, _) = t_loss_bkwd3_t(params, y, y_mask, y_indices, train=True, p_gen_aux = sample_p_gen_aux(params))
    
    # ugly..
    import sys
    sys.exit(0)


###############################################
### Infra utils
###############################################
def print_mem_stats():
    mem_stats = jax.devices()[0].memory_stats()
    conv = lambda k: mem_stats[k] / pow(1000,3)
    print(f'GB in use: {conv("bytes_in_use")}. GB limit: {conv("bytes_limit")}')

# start a new wandb run to track this script
if True:
    import wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="t_final",
    
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
import numpy as np # should we get rid of it?
import math

# Infra training params
run_name = datetime.datetime.now().strftime("%h%d_%H-%M-%S")
log_every_steps_multidevice = 10
eval_every_steps_multidevice = 1 if args.eval_only else 500 # clumsy support for eval only
eval_n_examples = 4
writer = SummaryWriter(f'/lego/storage/output/runs/{run_name}')
#checkpoint_every_steps = None #500 * 8 machines
checkpoint_every_steps = 4000 #20000 # TODO: move to use _multidevice too


# ML training params 
batch_size= 8 # 16 TODO XXX: WHY IT's not working for 16 like in JAX? 
gradient_accumulations_steps = 32 #16 -is for JAX # TODO XXX: This means effective batch_size=256 instead of 512 used in the paper
num_steps_multidevice = 10000 #100000 # TODO XXX: think what it should be for GPT2
max_lr = 0.00025
warmup_steps_multidevice= 2000
betas = (0.9, 0.98) 
epsilon = 10e-9
grads, moments = init_adam_w(params)

# TODO XXX: restructure, so we don't need below ones, and then, remove it
# NB: we use it later for inference/sampling, but the mask might be padded here leading to incomplete sampling.
_, _, _, y_eval_mask, _, _, y_eval_indices  = next(get_batched_examples(ds, eval_n_examples, seq_len, START_TOK, END_TOK, "validation")) 
    
i = 0 
ds_train_rows_read = 0
# TODO XXX: does "from_checkpoint" reproduces original training?
# It looks like it works fine for evals (i.e. it can reproduce the eval)
if args.from_checkpoint is not None: 
    with open(args.from_checkpoint,'rb') as f:
        i, ds_train_rows_read, params, moments = pickle.load(f)   
        print(f'Resuming training from the checkpoint: i {i} ds_train_rows_read {ds_train_rows_read}')

print(f'Number of params: {count_num_params(params):_}')

num_steps = num_steps_multidevice * gradient_accumulations_steps
while True:
    #for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0):
    for _, batch in tqdm(enumerate(itertools.islice(get_batched_examples_packed(ds, batch_size, seq_len, START_TOK, END_TOK, pack_frac=0.75, skip_n_rows = ds_train_rows_read), num_steps)), initial=i, total=num_steps, smoothing=0.3):
        _, y, _, y_mask, _, _, y_indices = batch
        # Training step
        # TODO: introduce update func, which does grad_loss and adam, and then call/jit that function instead of calling/jitting two separate ones
        # TODO XXX: int32 for y? we could use uint16 if it were available
        grads, (loss_val, acc, _) = acc_grad_loss_func(grads, params, torch.tensor(y, dtype=torch.int32, device="cuda"), torch.tensor(y_mask, dtype=torch.bool, device="cuda"), torch.tensor(y_indices, dtype=torch.int, device="cuda"))
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
        # NB: clumsy support for eval only. Do dummy training pass without updating params
        if i > 0 and i % gradient_accumulations_steps == 0 and not args.eval_only: 
            for grp_i in range(len(grads)):
                for p_i in range(len(grads[grp_i])):
                    #grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].divide(gradient_accumulations_steps) #TODO XXX: possible in place operation
                    grads[grp_i][p_i] = grads[grp_i][p_i]/gradient_accumulations_steps
            
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
                    #grads[grp_i][p_i] =  grads[grp_i][p_i].at[:].set(0) # TODO XXX: do it in place
                    grads[grp_i][p_i] =  torch.zeros_like(grads[grp_i][p_i], device="cuda")
            
        # Evaluation
        if i_multidevice>0 and i_multidevice%eval_every_steps_multidevice==0 and is_i_device_zero:
            val_losses = []
            val_accs = []
            val_toks_props = []
            for eval_step, batch in enumerate(get_batched_examples(ds, batch_size, seq_len, START_TOK, END_TOK, split="validation")): 
                _, y, _, y_mask, _, _, y_indices = batch
                _, (loss_val, acc, toks_prop) = loss_eval(params, torch.tensor(y, dtype=torch.int32, device="cuda"), torch.tensor(y_mask, dtype=torch.bool, device="cuda"), torch.tensor(y_indices, dtype=torch.int, device="cuda"))
                val_losses.append(loss_val.cpu())
                val_accs.append(acc.cpu())
                val_toks_props.append(toks_prop.cpu())
            eval_loss = np.average(np.hstack(val_losses), weights = np.hstack(val_toks_props)).item()
            eval_acc = np.average(np.hstack(val_accs), weights = np.hstack(val_toks_props)).item()
            print(f'Evaluation at step {i_multidevice}: loss {eval_loss} acc {eval_acc}')
            writer.add_scalar('eval/loss', eval_loss, i_multidevice)
            writer.add_scalar('eval/acc', eval_acc, i_multidevice)
            
            # Few predictions TODO XXX: vary temperature -> diff samples
            y_sample = predict(params, torch.tensor(y_eval_mask, dtype=torch.bool, device="cuda"), torch.tensor(y_eval_indices, dtype=torch.int, device="cuda"), seq_len, START_TOK, END_TOK)
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
                    choice_log_probs = log_probs(params, torch.tensor(y, dtype=torch.int32, device="cuda"), torch.tensor(y_mask, dtype=torch.bool, device="cuda"), torch.tensor(y_indices, dtype=torch.int, device="cuda"))
                    choices_vals.append(choice_log_probs.cpu()) # TODO XXX: move to GPU
                choices_vals = np.array(choices_vals).transpose() # we want choice per column
                hellaswag_accs.extend(np.argmax(choices_vals, axis=1)==labels)
                   
            hellaswag_acc = sum(hellaswag_accs)/len(hellaswag_accs)
            print(f'HellaSwag score:', hellaswag_acc)
            writer.add_scalar('eval/hellaswag', hellaswag_acc, i_multidevice)
            
            if args.eval_only: # clumsy support for eval_only flag
                # ugly..
                import sys
                sys.exit(0)
                
        i = i + 1
        ds_train_rows_read = ds_train_rows_read + len(y)

        # Checkpointing (i, ds_train_rows_read, params, moments).
        # TODO XXX: I haven't used it for a while, and likely it's not working.. probably we can delete 
        if checkpoint_every_steps is not None and (i>0 and i%checkpoint_every_steps==0):
            import os
            training_state = (i, ds_train_rows_read, params, moments)
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
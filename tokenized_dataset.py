# (Tokenized) DATASET: load_dataset and load_tokenized_dataset

import datasets
from datasets import load_dataset
from tokenizer import *

def load_wmt14_dataset(streaming=False):
    # load dataset
    print(f'Loading dataset')
    ds = load_dataset("wmt14", "de-en", streaming=streaming, cache_dir="datasets/") # 5.4mln rows (set cache to permanent space)
    ds = ds.map(lambda item: {'x':item['translation']["en"], 'y':item['translation']["de"]}, remove_columns=["translation"], batched=False, num_proc=12)

    return ds
    
def load_tokenized_dataset():
    ds = load_wmt14_dataset()
    
    # Load tokenizer + map
    tokenizer_filename = "bpe_tokenizer_ds_train_all_merges_35k.pickle" # "bpe_tokenizer_ds_train_all_merges_10k.pickle"
    print(f'Loading tokenizer {tokenizer_filename}')
    (tokenize, detokenize), state = load_bpe_tokenizer(tokenizer_filename)
    def encode_batch_map_func(batch):
        batch_str = [ x for x in batch['x']] + [ y for y in batch['y']]
        encoded = [ toks for toks in tokenize(batch_str)]
        return {'x':encoded[:int(len(encoded)/2)], 'y': encoded[int(len(encoded)/2):]}
    # Regarding batch_size below, the op is likely becoming memory-bound with bigger batch_size.
    # TODO: it would be worth investigating it, but I am skiping it in interest of time
    print(f'Tokenizing dataset')
    # TODO XXX: this sometimes hangs because of waiting for procs, should we reduce num_proc if loading from cache anyway?
    ds = ds.map(encode_batch_map_func, batched=True, num_proc=12, batch_size=50) 
    return ds, (tokenize, detokenize, len(state[0][0])) # dataset, tokenizer_state (..,.., vocab_len)

def _pad_xy_batch(batch, pad_length):
    def pad_tokens_list(tokens_list):
        pad = lambda toks: toks + [0] * (pad_length - len(toks))
        return [pad(toks)[:pad_length] for toks in tokens_list]
    return map(lambda x: pad_tokens_list(x), map(list, zip(*batch)))
    
def get_batched_examples(ds, batch_size, seq_len, split="train", skip_n_rows=None):
    ds_split = ds[split].skip(skip_n_rows) if skip_n_rows is not None else ds[split]
    
    batch = []
    for item in ds_split:
        batch.append((item['x'],item['y']))
        if len(batch) == batch_size:
            yield _pad_xy_batch(batch, seq_len)
            batch = []
    # TODO XXX: note I don't use few last rows left in train split
    if split!="train" and len(batch) > 0:
        # Adds the sequences consisting of pad tokens only
        # so batch_size is consistent, and vmap doesn't do more caching
        for _ in range(batch_size - len(batch)):
            batch.append(([0]*seq_len,[0] * seq_len))
        yield _pad_xy_batch(batch, seq_len)

def get_batched_examples_packed(ds, batch_size, seq_len, start_tok, end_tok, pack_frac = 0.5, split="train", skip_n_rows=None):
    ds_split = ds[split].skip(skip_n_rows) if skip_n_rows is not None else ds[split]
    
    batch = []
    for item in ds_split:
        # Either append to previous batch item or create new batch item
        if split=="train" and len(batch)>0 and (len(batch[-1][0]) < seq_len * pack_frac and len(batch[-1][1]) < seq_len * pack_frac):
            pack_func = lambda x, y: x + [end_tok, start_tok] + y
            batch[-1] = (pack_func(batch[-1][0], item['x']), pack_func(batch[-1][1], item['y']))
        else:
            batch.append((item['x'],item['y']))
            
        if len(batch) == batch_size:
            yield _pad_xy_batch(batch, seq_len)
            batch = []
            
    # TODO XXX: note I don't use few last rows left in train split
    if split!="train" and len(batch) > 0:
        # Adds the sequences consisting of pad tokens only
        # so batch_size is consistent, and vmap doesn't do more caching
        for _ in range(batch_size - len(batch)):
            batch.append(([0]*seq_len,[0] * seq_len))
        yield _pad_xy_batch(batch, seq_len)

def get_batched_examples_per_length(ds, total_tokens, split="train", approx_k=100):
    # Transformer's memory scales according to O(L^2 * N), where L is seq len, and N is batch size.
    # In order to choose batch size for different L without encountering OOMs later, we make the following heuristics:
    # 1) L * N <= total_tokens
    # 2) L^2 * N <= approx_k * total_tokens - this will only affect Ls bigger than approx_k i.e. beyond approx_k
    # i.e. For L<approx_k, N is inversely proportional to L. For L> approx_k, N is inversely proportial to L^2
    # TODO XXX: this is "quick and dirty", I should revist e.g., shouldn't 2) alone be enough? (First, make model more efficient though)
    
    # Use fixed bins for now
    max_length = 250 # TODO XXX: how many datapoints we truncate in training/eval?
    bin_width = 10
    batch_bins = [[] for _ in range(int(max_length/bin_width)+1)]
    
    for item in ds[split]:
        bin_i = (min(len(item['x']), max_length) +  bin_width - 1) // bin_width # ceil 
        batch_bins[bin_i].append((item['x'][:max_length],item['y'][:max_length]))
        bin_seq_len = bin_i * bin_width
        bin_total_tokens = len(batch_bins[bin_i]) * bin_seq_len
        if len(batch_bins[bin_i]) >= approx_k * total_tokens / pow(bin_seq_len, 2) or bin_total_tokens>total_tokens:
            yield _pad_xy_batch(batch_bins[bin_i], bin_i * bin_width)
            batch_bins[bin_i] = []
    for bin_i in range(len(batch_bins)):
        if len(batch_bins[bin_i]) > 0:
            yield _pad_xy_batch(batch_bins[bin_i], bin_i * bin_width)
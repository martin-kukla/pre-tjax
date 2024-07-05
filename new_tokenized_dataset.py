# (Tokenized) DATASET: load_dataset and load_tokenized_dataset

import datasets
from datasets import load_dataset
from tokenizer import *

###
# DATASET
###
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
    (tokenize, detokenize), state = load_bpe_tokenizer(f'tokenizers/{tokenizer_filename}')
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


###
# DATA COLLATOR
###

def build_masks(x, y, x_packs=None, y_packs=None, yx_packs=None): # x: seq_len, y: seq_len
    # Mask padded x tokens
    x_mask = np.ones((x.shape[0], x.shape[0]))
    x_pad_mask = np.where(x != 0, np.ones((x.shape[0])), 0)
    x_mask = np.multiply(np.multiply(x_mask, x_pad_mask), x_pad_mask[:, None])
    if x_packs is not None:
        x_mask = np.multiply(x_mask, x_packs)

    # Mask padded y tokens + add "autoregressive" mask
    y_pad_mask = np.where(y != 0, np.ones((y.shape[0])), 0)
    y_mask = np.tri(y.shape[0], y.shape[0])
    y_mask = np.multiply(np.multiply(y_mask, y_pad_mask), y_pad_mask[:, None])
    if y_packs is not None:
        y_mask = np.multiply(y_mask, y_packs)

    # Mask padded yx tokens
    yx_mask = np.ones((y.shape[0], x.shape[0]))
    yx_mask = np.multiply(np.multiply(yx_mask, x_pad_mask), y_pad_mask[:, None])
    if yx_packs is not None:
        yx_mask = np.multiply(yx_mask, yx_packs)

    return x_mask, y_mask, yx_mask

pad_and_trunc = lambda toks, seq_len: (toks + [0] * (seq_len - len(toks)))[:seq_len]
pack_batch = lambda batch: [np.stack(el) for el in map(list, zip(*batch))]

# Complete batch to batch_size with pad tokens only sequences 
def complete_last_batch(batch, batch_size):
    for _ in range(batch_size - len(batch)):
        x = np.zeros(seq_len)
        y = np.zeros(seq_len)
        x_mask, y_mask, yx_mask = build_masks(x,y)
        batch.append((x, y, x_mask, y_mask, yx_mask))
    return batch

def ones_block_diag(lens_dim0, lens_dim1):
    result = np.zeros((sum(lens_dim0), sum(lens_dim1)))
    start_ind_dim0 = 0
    start_ind_dim1 = 0
    for len_dim0, len_dim1 in zip(lens_dim0, lens_dim1):
        result[start_ind_dim0:start_ind_dim0+len_dim0, start_ind_dim1:start_ind_dim1+len_dim1] = np.ones((len_dim0, len_dim1))
        start_ind_dim0 +=len_dim0
        start_ind_dim1 +=len_dim1
    return result

def create_packs(x_lens, y_lens):
    x_packs = ones_block_diag(x_lens, x_lens)
    y_packs = ones_block_diag(y_lens, y_lens)
    yx_packs = ones_block_diag(y_lens, x_lens)
    return x_packs, y_packs, yx_packs

def create_indices(lens):
    return np.concatenate([np.arange(el_len) for el_len in lens])
    
def convert_batch_item(x, y, seq_len, x_lens=None, y_lens=None):
    x = np.array(pad_and_trunc(x, seq_len))
    y_plus_one = np.array(pad_and_trunc(y, seq_len + 1)) # We need "+1" version for training
    y = y_plus_one[:-1]

    # Account for packing
    if x_lens is not None:
        assert y_lens is not None
        # Trunc ?_lens up to seq_len
        def trunc_lens(lens):
            if sum(lens)>seq_len:
                lens[-1] = seq_len - sum(lens[:-1])
            else: # TODO: untested
                lens.append(seq_len - sum(lens))
            return lens
        x_lens = trunc_lens(x_lens)
        y_lens = trunc_lens(y_lens)
        x_packs, y_packs, yx_packs = create_packs(x_lens, y_lens)
        x_indices, y_indices = create_indices(x_lens), create_indices(y_lens)
    else:
        x_packs, y_packs, yx_packs, x_indices, y_indices = None, None, None, None, None
        
    x_mask, y_mask, yx_mask = build_masks(x,y, x_packs, y_packs, yx_packs)
    return (x, y_plus_one, x_mask, y_mask, yx_mask, x_indices, y_indices)
                
def get_batched_examples(ds, batch_size, seq_len, start_tok, end_tok, split="train", skip_n_rows=None):
    ds_split = ds[split].skip(skip_n_rows) if skip_n_rows is not None else ds[split]    

    batch = []
    for item in ds_split:
        item_y = [START_TOK] + item['y'] + [END_TOK]
        batch.append(convert_batch_item(item['x'], item_y, seq_len))
        if len(batch) == batch_size:
            yield pack_batch(batch)
            batch = []
    if split!="train" and len(batch) > 0: # Note I don't use last few rows left in train split..
        yield pack_batch(complete_last_batch(batch, batch_size))

def get_batched_examples_packed(ds, batch_size, seq_len, start_tok, end_tok, pack_frac = 0.5, split="train", skip_n_rows=None):
    assert split=="train"
    ds_split = ds[split].skip(skip_n_rows) if skip_n_rows is not None else ds[split]
    
    batch = []
    batch_x_lens = []
    batch_y_lens = []
    for item in ds_split:
        item_y = [START_TOK] + item['y'] + [END_TOK]
        
        # Either append to previous batch item or create new one
        if len(batch)>0 and (len(batch[-1][0]) < seq_len * pack_frac and len(batch[-1][1]) < seq_len * pack_frac):
            batch[-1] = (batch[-1][0] + item['x'], batch[-1][1] +  item_y)
            batch_x_lens[-1].append(len(item['x']))
            batch_y_lens[-1].append(len(item_y))
        else:
            batch.append((item['x'],item_y))
            batch_x_lens.append([len(item['x'])])
            batch_y_lens.append([len(item_y)])
            
        if len(batch) == batch_size:
            print(batch_x_lens)
            print(batch_y_lens)
            batch = [convert_batch_item(*it, seq_len, x_lens, y_lens) for it, x_lens, y_lens in zip(batch, batch_x_lens, batch_y_lens)]
            
            yield pack_batch(batch)
            batch = []
            
    if split!="train" and len(batch) > 0: # Note I don't use last few rows left in train split..
        yield pack_batch(complete_last_batch(batch, batch_size))
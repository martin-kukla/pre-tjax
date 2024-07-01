import pickle
import heapq
from tqdm import tqdm


def get_words(str): # TODO: Note we only work on lower-case, but how does one go to the orignal string?
    return str.lower().split()

def build_w_vocab(ds):
    vocab = {}
    for item in tqdm(ds):
        for w in get_words(item['x']) + get_words(item['y']):
            vocab[w] = vocab.get(w, 0) + 1
    print(f'len(vocab) {len(vocab)}')
    return vocab


### WORD TOKENIZER

def make_w_tokenizer_func(state):
    def w_tokenize(vocab_map, str):
        return [vocab_map[s] for s in get_words(str)]
        
    def w_detokenize(reversed_vocab_map, toks):
        return " ".join([reversed_vocab_map[tok] for tok in toks if tok!=0])


    return (lambda s : w_tokenize(state[0], s), lambda toks:w_detokenize(state[1], toks))
    
def build_w_tokenizer(ds):
    vocab = build_w_vocab(ds)
    
    # Functions for (de-)tokenization
    vocab_map = {w: (i+1) for i, w in enumerate(vocab.keys())} # shift by one: 0 reserved for padding
    reversed_vocab_map = {v:k for k,v in vocab_map.items()}
    state = (vocab_map, reversed_vocab_map)

    return make_w_tokenizer_func(state), state

def save_tokenizer(state, path):
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def load_tokenizer_state(path):
    with open(path, 'rb') as f:
         return pickle.load(f)

def load_w_tokenizer(path):
    state = load_tokenizer_state(path)
    return make_w_tokenizer_func(state), state


### BPE tokenizer

from operator import itemgetter
import re

def bpeify_w(w):
    return " ".join(w) + ' </w>'

def merge_w(w, pair):
    s_in = re.escape(" ".join(pair))
    return re.sub(f'(?<!\S){s_in}(?!\S)', "".join(pair), w)
        
def make_bpe_tokenizer_func(state):

    def apply_bpe_merges(bpe_merges, bpe_str):
        bpe_str = " \t ".join(bpe_str) # overload "\t", so I can apply a merge to all words in sequence at once
        for bpe_merge in bpe_merges:
            bpe_str = merge_w(bpe_str, bpe_merge)
        return bpe_str.split(" \t ")
    
    def bpe_tokenize(state, str): # TODO speedup: precompute map: a word -> list of bpe tokens]
        bpe_vocab_map, bpe_merges = state
        bpe_str = [bpeify_w(w) for w in get_words(str)]
        bpe_str = apply_bpe_merges(bpe_merges, bpe_str)   
    
        def bpe_tokenize_w(w):
            return [bpe_vocab_map[s] for s in w.split(" ")]
    
        return [t for w in bpe_str for t in bpe_tokenize_w(w)]

    # apply ALL bpe_merges no matter input str
    def batched_apply_bpe_merges(bpe_merges, batched_bpe_str):
        # overload whitespaces, so I can apply a merge to all words in the batch at once
        flattened_bpe_str = " \r ".join([" \t ".join(bpe_str) for bpe_str in batched_bpe_str]) 
        for bpe_merge in bpe_merges:
            flattened_bpe_str = merge_w(flattened_bpe_str, bpe_merge)
        return [bpe_str.split(" \t ") for bpe_str in flattened_bpe_str.split(" \r ")]

    # iterate over input string to find present bpe_merges, and then apply only those
    def batched_apply_bpe_merges2(merges_dict, batched_bpe_str):
        # overload whitespaces, so I can apply a merge to all words in the batch at once
        flattened_bpe_str = " \r ".join([" \t ".join(bpe_str) for bpe_str in batched_bpe_str]) 
        while True:
            w = flattened_bpe_str.split(" ")
            best_pair = None
            best_score = len(state[0][1]) + 1
            for i, sym in  enumerate(w[:-1]): # TODO: update this incrementally!
                pair = (sym, w[i+1])
                if pair in merges_dict and merges_dict[pair]< best_score:
                    best_pair = pair
                    best_score = merges_dict[pair]
            if best_pair is None:
                break
            flattened_bpe_str = merge_w(flattened_bpe_str, best_pair)
    
        return [bpe_str.split(" \t ") for bpe_str in flattened_bpe_str.split(" \r ")]

    # Incremental implementation: after applying a merge, figure out next merge without iterating over string
    # Data structures: 
    # dictionary of present merges +  priority queue + pointers (not indices) for occurences stored as double linked list
    def batched_apply_bpe_merges4(merges_dict, batched_bpe_str):
        # Doubly linked list
        class Node():
            def __init__(self, sym):
                self.sym = sym
                self.next = None
                self.prev = None
            def __repr__(self):
                return self.sym
                
        class DLList():
            def __init__(self, head):
                self.head = head
    
            def get_str(self):
                node = w.head
                str = None
                while node is not None:
                    str = node.sym if str is None else str + " " + node.sym
                    node = node.next
                return str
             
            def from_str(flattened_bpe_str):
                head_node = None
                previous_node = None
                for w in flattened_bpe_str.split(" "):
                    if head_node is None:
                        head_node = Node(w)
                        previous_node = head_node
                        continue
        
                    node = Node(w)
                    node.prev = previous_node
                    previous_node.next = node
                    previous_node = node
        
                return DLList(head_node)
        
        # pair occurences + priority queue
        def add_pair_occ(pair, present_merges, node, priority_queue):
            if pair in merges_dict:
                if pair not in present_merges:
                    present_merges[pair] = (merges_dict[pair], [node])
                    heapq.heappush(priority_queue, (merges_dict[pair], pair))
                else:
                    present_merges[pair] =  (present_merges[pair][0], present_merges[pair][1] +[node])
        def remove_pair_occ(pair, present_merges, node):
            if pair in present_merges:
                present_merges[pair][1].remove(node)
                if len(present_merges[pair][1]) == 0:
                    del present_merges[pair]
        
        # overload whitespaces, so I can apply a merge to all words in the batch at once
        flattened_bpe_str = " \r ".join([" \t ".join(bpe_str) for bpe_str in batched_bpe_str]) 
    
        # create double linked list, and index all merges initially present
        w = DLList.from_str(flattened_bpe_str) # TODO: move code here instead
        node = w.head
        present_merges = {} # pair -> (score, occurences)
        priority_queue = [] # (score, pair) - as heapq sorts by 1st element of tuple
        while node is not None:
            if node.next is None:
                break
            add_pair_occ((node.sym, node.next.sym), present_merges, node, priority_queue) # TODO: pass both nodes
            node = node.next
        
        # Incrementally apply present merges according to their priority
        while len(priority_queue)>0:
            score, best_pair = heapq.heappop(priority_queue)
            if best_pair not in present_merges:
                continue
            nodes = present_merges[best_pair][1]
            #del present_merges[best_pair]
            new_sym = best_pair[0] + best_pair[1]
    
            # Apply merge to all occurences
            previous_right_node = None
            for left_node in nodes:
                right_node = left_node.next

                # We need to account for a merge occuring twice in a row
                # e.g. for "9 9 9", and merge "9 9", we only want to merge "9 9"
                if previous_right_node is not None:
                    if left_node == previous_right_node:
                        previous_right_node = None
                        continue
                previous_right_node = right_node
    
                # remove occurences of old nodes (left and right)
                if left_node.prev is not None:
                    remove_pair_occ((left_node.prev.sym, left_node.sym), present_merges, left_node.prev)
                if right_node.next is not None:
                    remove_pair_occ((right_node.sym, right_node.next.sym), present_merges, right_node)
    
                # merge nodes
                new_node = Node(new_sym)
                if left_node.prev is not None:
                    new_node.prev = left_node.prev
                    left_node.prev.next = new_node
                if right_node.next is not None:
                    new_node.next = right_node.next
                    right_node.next.prev = new_node
                if left_node==w.head:
                    w.head = new_node
    
                # add occurences of new node
                if new_node.prev is not None:
                    add_pair_occ((new_node.prev.sym, new_sym), present_merges, new_node.prev, priority_queue)
                if new_node.next is not None:
                    add_pair_occ((new_sym, new_node.next.sym), present_merges, new_node, priority_queue)

            del present_merges[best_pair]
    
        flattened_bpe_str = w.get_str()
        return [bpe_str.split(" \t ") for bpe_str in flattened_bpe_str.split(" \r ")]

    def batched_bpe_tokenize(state, batched_str): 
        bpe_vocab_map, bpe_merges = state
        def bpeify_str(str):
            return [bpeify_w(w) for w in get_words(str)]
        batched_bpe_str = [bpeify_str(str) for str in batched_str]
        #batched_bpe_str = batched_apply_bpe_merges(bpe_merges, batched_bpe_str) 
        #batched_bpe_str = batched_apply_bpe_merges2(bpe_merges, batched_bpe_str)
        batched_bpe_str = batched_apply_bpe_merges4(bpe_merges, batched_bpe_str)
    
        def bpe_tokenize_w(w):
            return [bpe_vocab_map[s] for s in w.split(" ")]
        def bpe_tokenize_str(bpe_str):
            return [t for w in bpe_str for t in bpe_tokenize_w(w)]
            
        return [bpe_tokenize_str(bpe_str) for bpe_str in batched_bpe_str]
    
    def bpe_detokenize(reversed_bpe_vocab_map, toks):
        return "".join([reversed_bpe_vocab_map[tok] for tok in toks if tok!=0]).replace("</w>", " ")

    def batched_bpe_detokenize(reversed_bpe_vocab_map, batched_toks):
        return [bpe_detokenize(reversed_bpe_vocab_map, toks) for toks in batched_toks]

    # For batched_apply_bpe_merges
    #return (lambda batched_s : batched_bpe_tokenize(state[0], batched_s), lambda batched_toks:batched_bpe_detokenize(state[1], batched_toks))

    # For batched_apply_bpe_merges2
    state_with_merges_dict = (state[0][0], {pair: i for i, pair in enumerate(state[0][1])})
    return (lambda batched_s : batched_bpe_tokenize(state_with_merges_dict, batched_s), lambda batched_toks:batched_bpe_detokenize(state[1], batched_toks))
    
def get_pairs_w(w):
        w_symbols = w.split(" ")
        return [(curr_s, w_symbols[i+1]) for i, curr_s in enumerate(w_symbols[:-1])]
    
def establish_bpe_merges(bpe_vocab, num_merges):
    def get_pairs(vocab):
        pairs = {}
        for w, count in vocab.items():
            for pair in get_pairs_w(w):
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs
        
    def merge_vocab(vocab, pair):
        return {merge_w(w, pair):c for w, c in vocab.items()}
        
    bpe_merges = []
    for _ in tqdm(range(num_merges)):
        pairs = get_pairs(bpe_vocab)
        pair = max(pairs, key=pairs.get)
        bpe_merges.append(pair)
        bpe_vocab = merge_vocab(bpe_vocab, pair)

    return bpe_vocab, bpe_merges

# Establish bpe merges incrementally. Two main commments:
# a) the implementation can be cleaned up
# b) data structures & logic is different than when we apply merges during "inference" (batched_apply_bpe_merges4).
#     If we unite both, we will probably get cleaner code (+ speedups) (BIG TODO)
#
# This implementation uses the following strucutres:
# 1. bpe_vocab: word -> word_count
# 2. pairs_to_vocab: pair -> [(word, occurences_of_pair_in_word)]
#    pairs_to_count: pair -> total_pair_occurences (in bpe_vocab)
# 3. vocab_to_pairs: word -> [pair]
# It's incremental, since, instead of computing the pairs present across vocab,
# we compute these pairs once, and then update the structures (1-3)
#
# There are two lines which can be done faster, but it works fast enough for needs of 
# replicating results from "Attention is All You Need" paper.
def establish_bpe_merges4(bpe_vocab, num_merges):
    def get_pairs(vocab_words):
        pairs_to_vocab = {}
        vocab_to_pairs = {}
        for w in vocab_words:
            w_pairs = get_pairs_w(w)
            vocab_to_pairs[w] = w_pairs
            for pair in w_pairs:
                if pair not in pairs_to_vocab:
                    pairs_to_vocab[pair] = {}
                pairs_to_vocab[pair][w] = pairs_to_vocab[pair].get(w, 0) + 1
        return pairs_to_vocab, vocab_to_pairs

    # Efficient implementation: O(N+M), where N is len(old_w_pairs), and M is len(new_w_Pairs)
    # TODO: clean up the code
    def diff_w_pairs(old_w_pairs, new_w_pairs, new_merged_token): 
        old_w_old_pairs = []
        unchanged_w_pairs = []
        new_w_new_pairs = []
        i = 0
        j = 0
        
        while i < len(old_w_pairs) and j < len(new_w_pairs):
            if old_w_pairs[i] == new_w_pairs[j]:
                unchanged_w_pairs.append(new_w_pairs[j])
                i = i + 1
                j = j + 1
            else: # there was a merge
                while j < len(new_w_pairs) and new_merged_token in new_w_pairs[j]:
                    new_w_new_pairs.append(new_w_pairs[j])
                    j = j+1
                while i < len(old_w_pairs) and (not j< len(new_w_pairs) or old_w_pairs[i] != new_w_pairs[j]):
                    old_w_old_pairs.append(old_w_pairs[i])
                    i = i+1

        return old_w_old_pairs, unchanged_w_pairs, new_w_new_pairs
    
    bpe_merges = []
    
    pairs_to_vocab, vocab_to_pairs = get_pairs(bpe_vocab.keys())
    def sum_vocab_dict(vocab_dict):
        return sum(bpe_vocab[w] * w_c for w, w_c in vocab_dict.items())
    pairs_to_count = {pair: sum_vocab_dict(v) for pair, v in pairs_to_vocab.items()}

    for i in tqdm(range(num_merges)):  
        pair = max(pairs_to_count, key=lambda pair: pairs_to_count[pair])
        bpe_merges.append(pair)

        # incrementally apply the merge
        new_vocab_words = []
        old_words = set(pairs_to_vocab[pair])
        for old_w in old_words: 
            new_w = merge_w(old_w, pair)
            new_w_pairs = get_pairs_w(new_w)
            w_count = bpe_vocab.pop(old_w)
            bpe_vocab[new_w] = w_count
            new_vocab_words.append(new_w)

            # incremetally update pairs_to_vocab struct
            old_w_pairs = vocab_to_pairs[old_w]
            # TODO: expensive part, we can probably expand merge_w to give us these structs (merge_w is done through compiled regex though..)
            old_w_old_pairs, w_unchaged_w_pairs, new_w_new_pairs = diff_w_pairs(old_w_pairs, new_w_pairs, pair[0]+pair[1])
            for old_pair in old_w_old_pairs:
                pairs_to_vocab[old_pair][old_w] = pairs_to_vocab[old_pair][old_w] - 1
                pairs_to_count[old_pair] = pairs_to_count[old_pair] - w_count
                if pairs_to_vocab[old_pair][old_w] == 0:
                    del pairs_to_vocab[old_pair][old_w]
            # Expensive part, which we can skip. TODO: add indirection layer?
            # pairs_to_vocab_ind and vocab_ind_to_vocab, so we will not need to update pairs_to_vocab_ind
            for unchanged_w_pair in w_unchaged_w_pairs: 
                pairs_to_vocab[unchanged_w_pair][new_w] = pairs_to_vocab[unchanged_w_pair].get(new_w, 0) + 1
                pairs_to_vocab[unchanged_w_pair][old_w] = pairs_to_vocab[unchanged_w_pair][old_w] - 1
                # pairs_to_counts doesn't change!
                if pairs_to_vocab[unchanged_w_pair][old_w] == 0:
                    del pairs_to_vocab[unchanged_w_pair][old_w]
            for new_pair in new_w_new_pairs: # The idea is that this set should be much smaller than new_w_pairs
                if new_pair not in pairs_to_vocab:
                    assert new_pair not in pairs_to_count
                    pairs_to_vocab[new_pair] = {}
                pairs_to_vocab[new_pair][new_w] = pairs_to_vocab[new_pair].get(new_w, 0) + 1
                pairs_to_count[new_pair] = pairs_to_count.get(new_pair, 0) + w_count

            # incrementally update vocab_to_pairs struct
            del vocab_to_pairs[old_w]
            vocab_to_pairs[new_w] = new_w_pairs
        del pairs_to_vocab[pair]
        del pairs_to_count[pair]

    return bpe_vocab, bpe_merges
    
def build_bpe_tokenizer(ds, num_merges=10):
    print(f'Build BPE-ified word vocabulary')
    input_vocab = build_w_vocab(ds)
    bpe_vocab = {bpeify_w(w):c for w,c in input_vocab.items()}

    print(f'Establish {num_merges} BPE merges')
    bpe_vocab, bpe_merges = establish_bpe_merges4(bpe_vocab, num_merges)
    
    # Functions for (de-)tokenization
    all_initial_symbols = [ch for w in input_vocab.keys() for ch in w]
    all_bpe_symbols = [s for bpe_w in bpe_vocab.keys() for s in bpe_w.split(" ")]
    uniq_bpe_symbols = list(dict.fromkeys(all_initial_symbols + all_bpe_symbols))
    print(f'#uniq_bpe_symbols {len(uniq_bpe_symbols)}')
    bpe_vocab_map = {w: (i+1) for i, w in enumerate(uniq_bpe_symbols)} # shift by one: 0 reserved for padding
    reversed_bpe_vocab_map = {v:k for k,v in bpe_vocab_map.items()}

    state = ((bpe_vocab_map, bpe_merges), reversed_bpe_vocab_map)
    return make_bpe_tokenizer_func(state), state

def load_bpe_tokenizer(path):
    state = load_tokenizer_state(path)
    return make_bpe_tokenizer_func(state), state

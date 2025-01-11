import argparse
import random
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


class TokenPermuter:
    """Permutes the token ids of an NLLB tokenizer."""
    
    def __init__(self, permutation):    
        self.permutation = permutation

    def map_token_id(self, token_id):
        if token_id > 256000: # tokens above this are language ids
            return 3          # UNK token
        elif token_id not in self.permutation:  # if the token id isn't in the permutation, it maps to itself
            return token_id
        else:
            return self.permutation[token_id]
                

def create_token_permuter(tokenizer, sents):
    def permute(indexed_tokens):        
        indices = [tok[1] for tok in indexed_tokens]
        permutation = [tok[1] for tok in indexed_tokens]
        random.shuffle(permutation)
        return dict(zip(indices, permutation))
    
    tokenized = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=128) 
    ids = [idx for idx in tokenized['input_ids'].unique().tolist() if 4 <= idx <= 256000] # specific to NLLB model
    tokens = [(tokenizer.convert_ids_to_tokens(next_id), next_id) for next_id in ids]
    length1_tokens = [tok for tok in tokens if len(tok[0]) == 1]
    other_tokens = [tok for tok in tokens if len(tok[0]) > 1]
    underscore_tokens = [tok for tok in other_tokens if tok[0].startswith("▁")]
    no_underscore_tokens = [tok for tok in other_tokens if not tok[0].startswith("▁")]        
    vocab = permute(underscore_tokens)
    vocab.update(permute(no_underscore_tokens))
    vocab.update(permute(length1_tokens))
    return TokenPermuter(vocab)


def encrypt_sentences(sents, tokenizer, permuter):
    tokenized = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=128)
    original_ids = tokenized['input_ids'].clone()
    original_ids.apply_(permuter.map_token_id)
    encrypted = tokenizer.batch_decode(original_ids, skip_special_tokens=True)
    return encrypted


def choose_split(i):
    if i > 2000:
        return 'train'
    elif i > 1000:
        return 'dev'
    else:
        return 'test'

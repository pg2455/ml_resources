import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence 

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

import nltk
import math
import numpy as np
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt 
import pathlib

# fix seed for reproducibility 
rng = np.random.RandomState(1)
torch.manual_seed(rng.randint(np.iinfo(int).max))

# it is a good practice to define `device` globally
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", device)
else:
    device = torch.device("cpu")
    print("No GPU -> using CPU:", device)

LANGUAGE_PAIRS = ['de', 'en']
Multi30k(root="./data", split=("train", "valid", "test"), language_pair=LANGUAGE_PAIRS)


def get_data_iter(itertype):
    return Multi30k(root="./data", split=itertype, language_pair=LANGUAGE_PAIRS)



SRC_LANGUAGE = LANGUAGE_PAIRS[0]
TGT_LANGUAGE = LANGUAGE_PAIRS[1]

print(f'Source language: {SRC_LANGUAGE}\t Target language: {TGT_LANGUAGE} ')

SPACY_LANGUAGE_MAP = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm'
}

## tokenization
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language=SPACY_LANGUAGE_MAP[SRC_LANGUAGE])
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language=SPACY_LANGUAGE_MAP[TGT_LANGUAGE])

## token to index

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def yield_tokens(data_iter, language):
    idx = LANGUAGE_PAIRS.index(language)
    
    for sentence in data_iter:
        yield token_transform[language](sentence[idx])

vocab_transform = {}
vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(yield_tokens(get_data_iter("train"), SRC_LANGUAGE), min_freq=1, specials=special_symbols, special_first=True)
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(yield_tokens(get_data_iter("train"), TGT_LANGUAGE), min_freq=1, specials=special_symbols, special_first=True)

for v in vocab_transform.values():
    v.set_default_index(UNK_IDX)

## marking beginning and end of a sentence
def bos_eos_pad(tokens):
    return torch.cat((
            torch.tensor([BOS_IDX]), 
            torch.tensor(tokens), 
            torch.tensor([EOS_IDX])
    ))

## Putting it all together
def get_text_transform(language):
    def func(sentence):
        x = token_transform[language](sentence)
        x = vocab_transform[language](x)
        x = bos_eos_pad(x)
        return x
    return func

text_transform = {}
text_transform[SRC_LANGUAGE] = get_text_transform(SRC_LANGUAGE)
text_transform[TGT_LANGUAGE] = get_text_transform(TGT_LANGUAGE)

# n tokens
SRC_N_TOKENS = len(vocab_transform[SRC_LANGUAGE])
TGT_N_TOKENS = len(vocab_transform[TGT_LANGUAGE])

# from https://pytorch.org/tutorials/beginner/translation_transformer.html#collation

# function to collate data samples into batch tesors
def collate_fn(batch):
    """
    Args:
        batch (iterator): each element is a source sentence and target sentence 
    
    Returns:
        src_batch (torch.tensor): [n_sentences x max_sentence_lenght]
        tgt_batch (torch.tensor): [n_sentences x max_sentence_lenght]
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch.permute(1,0), tgt_batch.permute(1,0)

def get_sample_batch(type, n):
    data_iter = get_data_iter(type)
    batch = [next(data_iter) for _ in range(n)]
    src, tgt = collate_fn(batch)
    return src, tgt

def get_readable_tokens(tokens, language):
    assert len(tokens.shape) == 1, f"unrecognized input shape: {tokens.shape}"
    return vocab_transform[language].lookup_tokens(tokens.numpy())
    
def get_readable_text(readable_tokens):
    return " ".join(readable_tokens).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")


def mem_size(model):
    """
    Get model size in GB (as str: "N GB")
    """
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return f"{mem / 1e9:.4f} GB"

def num_params(model):
    """
    Print number of parameters in model's named children
    and total
    """
    s = "Number of parameters:\n"
    n_params = 0
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        s += f"  • {name:<15}: {n}\n"
        n_params += n
    s += f"{'total':<19}: {n_params}"

    return s

def pp_model_summary(model):
    print(num_params(model))
    print(f"{'Total memory':<18} : {mem_size(model)}")
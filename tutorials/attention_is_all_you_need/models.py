import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence 

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

## token to index

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def attention(Q, K, V, mask=None, dropout=None):
    """
    Implements attention mechanism.
    
    Args:
        Q (torch.tensor): [batch_size x heads x max_sentence_length x dimension] Query matrix 
        K (torch.tensor): [batch_size x heads x max_sentence_length x dimension] Key matrix 
        V (torch.tensor): [batch_size x heads x max_sentence_length x dimension] Value matrix 
        mask (torch.tensor): [max_sentence_length x max_sentence_length] mask to prevent certain queries attending to certain keys
        dropout (F.dropout): 
    
    Returns:
        (torch.tensor): Convex combination of V where weights are decided by the attention mechanism
    """
    d_k = K.shape[-1]
    # bs x n_heads x max_sentence_length x max_sentence_length
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, V)
    
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1): 
        
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        
        batch_size = q.shape[0]
        
        # break the output such that the last two dimensions are heads x d_head
        Q = self.Q(q).view(batch_size, -1, self.n_heads, self.d_head)
        K = self.K(k).view(batch_size, -1, self.n_heads, self.d_head)
        V = self.V(v).view(batch_size, -1, self.n_heads, self.d_head)
        
        # take transpose so that last two dimensions are max_sentence_length x d_head (as required for the attention module)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # batch_size x n_heads x max_sentence_length x d_head
        attn, scores = attention(Q, K, V, mask, self.dropout)
        
        # transpose back to get batch_size x sentence_length x d_model
        # use contiguous to reset the ordering of elements (i.e. stride and offset): https://stackoverflow.com/a/52229694/3413239
        concat = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return concat, scores
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1):
        
        super().__init__()
        
        # mha
        self.mha = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # dropout regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        z, s = self.mha(x, x, x, mask) # mha
        z = self.dropout(z) # refer to the section 5.4 of Vaswani et al. 
        x = self.norm1(x + z) # add & norm
        
        z = self.ff(x) # feed forward
        z = self.dropout(z) # refer to the section 5.4 of Vaswani et al. 
        x = self.norm2(x + z) # add & norm
        return x, s


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1):
        
        super().__init__()
        
        # masked-mha
        self.mha1 = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # mha
        self.mha2 = MultiHeadAttention(n_heads, d_model)
        self.norm2 = nn.LayerNorm(d_model)
                
        # feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        # dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, e_x, src_mask, tgt_mask):
        z, s = self.mha1(x, x, x, tgt_mask) # masked mha 
        z = self.dropout(z) # refer to the section 5.4 of Vaswani et al. 
        x = self.norm1(x + z) # add & norm 
        
        z, cross_attn_scores = self.mha2(x, e_x, e_x, src_mask) # mha 
        z = self.dropout(z) # refer to the section 5.4 of Vaswani et al. 
        x = self.norm2(x + z) # add & norm 
        
        z = self.ff(x)
        z = self.dropout(z) # refer to the section 5.4 of Vaswani et al. 
        x = self.norm3(x + z)
        
        return x, s, cross_attn_scores 


class TokenEmbedding(nn.Module):
    def __init__(self, n_tokens, emb_size):
        super().__init__()
        
        self.embedding = nn.Embedding(n_tokens, emb_size)
        self.emb_size = emb_size 
    
    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def get_position_encodings(max_len, d):
    """
    Creates encoding
    
    Args:
        max_len (int): number of positions to encode
        d (int): dimension of encoding
    
    Returns:
        (torch.tensor): [max_len, d] each row is the encoding for that row position
    """
    exp_coeff = torch.arange(0, d, 2) / d
    arg = torch.exp(-exp_coeff * math.log(10000)) # take exp of log for numerical stability
    
    pos = torch.arange(0, max_len).reshape(max_len, 1)
    pos_embedding = torch.zeros((max_len, d))
    pos_embedding[:, 0::2] = torch.sin(pos * arg)
    pos_embedding[:, 1::2] = torch.cos(pos * arg)
    return pos_embedding


def create_masks(src, tgt):
    src_padding_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    
    N, L = tgt.shape
    tgt_mask = torch.tril(torch.ones((L, L)), diagonal=0).expand(N, 1, L, L)
    
    return src_padding_mask, tgt_mask


class Transformer(nn.Module):
    """
    Args:
       src_n_tokens (int): vocabulary size or number of unique tokens in the source language
       tgt_n_tokens (int): vocabulary size or number of unique tokens in the target language
       n_encoders (int): number of encoders to stack on top of each other
       n_decoders (int): number of decoders to stack on top of each other
       d_model (int): embedding dimension size
       n_heads (int): number of heads in multi-head attention
       d_ff (int): hidden dimension of feed forward networks
       dropout (float): dropout probability
       max_length (int): maximum length for positional encodings (optional).
    """
    def __init__(self, src_n_tokens, tgt_n_tokens, n_encoders, n_decoders,  d_model, n_heads, d_ff=1024, dropout=0.1, max_length=1000):
        super().__init__()
        
        self.input_args = [src_n_tokens, tgt_n_tokens, n_encoders, n_decoders,  d_model, n_heads, d_ff, dropout, max_length]
        
        # embedddings + positional encoding
        self.src_embeddings = TokenEmbedding(src_n_tokens, d_model)
        self.tgt_embeddings = TokenEmbedding(tgt_n_tokens, d_model)
        self.register_buffer('positional_encodings', nn.Parameter(get_position_encodings(max_length, d_model)))
        
        # encoders
        self.encoders = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoders) 
        ])
        
        # decoders
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoders) 
        ])
        
        # final linear (pre-softmax)
        self.out = nn.Sequential(
            nn.Linear(d_model, tgt_n_tokens),
        )
        
        # dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_padding_mask, tgt_mask, src_embedding=None):
        # encode
        e_x, ats = self.encode(src, src_padding_mask, src_embedding)
        
        # decode 
        z, ss, cs = self.decode(tgt, e_x, src_padding_mask, tgt_mask)
        
        # linear (pre-softmax)
        out = self.out(z)
        
        return out, {'encoder_attn': ats, 'decoder_ss': ss, 'decoder_cs': cs}

    def encode(self, src, src_padding_mask, src_embedding=None):
        """
        Returns the encodings for src sentence
        """
        # encoder inputs
        if src_embedding is not None:
            src_emb = src_embedding
        else:
            src_emb = self.src_embeddings(src)

        src_emb = src_emb + self.positional_encodings[:src.shape[1], :].unsqueeze(0) 
        src_emb = self.dropout(src_emb) # dropout described in section 5.4 of Vaswani et al.
        
        # encoder
        attn_scores = []
        for encoder in self.encoders:
            e_x, s = encoder(src_emb, src_padding_mask)
            attn_scores.append(s)
        
        return e_x, attn_scores
    
    def decode(self, tgt, e_x, src_padding_mask, tgt_mask):
        """
        Returns the pre-softmax decoder outputs
        """
        # decoder inputs 
        tgt_emb = self.tgt_embeddings(tgt) + self.positional_encodings[:tgt.shape[1], :].unsqueeze(0) 
        tgt_emb = self.dropout(tgt_emb) # dropout described in section 5.4 of Vaswani et al.
        
        # decoder 
        self_scores, cross_scores = [], []
        z = tgt_emb
        for decoder in self.decoders:
            z, s, cs = decoder(z, e_x, src_padding_mask, tgt_mask)
            self_scores.append(s)
            cross_scores.append(cs)
        
        return z, self_scores, cross_scores

    
# adapted from https://stackoverflow.com/a/66773267/3413239

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, epsilon=0.0):
        super().__init__()
        self.smoothing = epsilon / (classes - 2) # classes to not consider for smoothing:  PAD_IDX class, target class
        self.confidence = 1 - epsilon 
    
    def forward(self, x, target):
        
        # mask out PAD_IDX
        tgt_padding_mask = (target != PAD_IDX).unsqueeze(2)
        tgt_padding_mask = tgt_padding_mask.to(x.device)
        x = x.masked_fill(tgt_padding_mask==0, 0)
    
        # compute CE loss on smoothed distribution
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        target_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(2)).squeeze(2)
        smooth_loss = -logprobs.sum(dim=-1)
        loss = (self.confidence - self.smoothing) * target_loss + self.smoothing * smooth_loss
        return loss.mean()


def greedy_decoding(model, src, max_len):
    """
    Implements greedy search for a most likely translation of `src`.
    
    Args:
        model (Transformer): model to extract p(y_t | y_{<t}, x)
        src (torch.tensor): [n_sentences x max_sentence_length] source sentence with paddings, BOS and EOS tokens (as expected by the model)
        max_len (int): maximum length of the translated sentence.
    
    Returns:
        (torch.tensor): indices of tokens in a target sentence
    """
    model.to(device)
    model.eval() # no use of dropout from here on
    
    src = src.to(device)
    N  = src.shape[0]  # n_sentences
    eos_tracker = set()
    with torch.no_grad():
        
        # src padding
        src_padding_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        src_padding_mask = src_padding_mask.to(device)
        
        # source encoding (x)
        e_x, _ = model.encode(src, src_padding_mask) # source encodings
        
        # output decoding (y_i)
        outputs = torch.empty((N, max_len + 1), dtype = torch.long)
        outputs[:] = PAD_IDX
        outputs = outputs.to(device)
        outputs[:, 0] = BOS_IDX
        
        for i in range(1, max_len+1):
            tgt_mask = torch.tril(torch.ones(N, 1, i, i), diagonal=0)
            tgt_mask = tgt_mask.to(device)
            
            out = model.out(model.decode(outputs[:, :i], e_x, src_padding_mask, tgt_mask)[0])
            ix = out[:, -1, :].data.topk(1).indices # last position's predictions only
            
            rem_sentences = [i not in eos_tracker for i in range(N)]
            outputs[rem_sentences, i] = ix[rem_sentences, 0]
            
            eos_tracker.update(set(torch.where(ix[:, 0] == EOS_IDX)[0].tolist())) # these sentences have finished
            
            if len(eos_tracker) == N:
                break
    
    return outputs.to('cpu')
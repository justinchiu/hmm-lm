import sys
import functools

from itertools import product

import numpy as np

import torch as th
import torch_struct as ts
import torch.nn as nn

from models.chmmlm import ChmmLm

from assign import assign_states, assign_states2, assign_states3
from utils import Pack

# word-state assignment
num_states = int(8) 
words_per_state = int(4) 
num_words = int(16) 
states_per_word = int(2)

num_states = int(64) 
words_per_state = int(32) 
num_words = int(128) 
states_per_word = int(8)

config = Pack(
    device = -1,
    old_res = True,
    num_layers = 1,
    hidden_dim = 256,
    emb_dim = 256,
    semiring = "FastLogSemiring",
    num_classes = num_states,
    words_per_state = words_per_state,
    states_per_word = states_per_word,
)
chmm = ChmmLm([x for x in range(num_words)], config)

"""
num_states = int(1024) 
words_per_state = int(512) 
num_words = int(2048) 
states_per_word = int(128)
"""

assert num_states * words_per_state >= num_words * states_per_word
#word2state, state2word = assign_states(num_states, states_per_word, num_words, words_per_state)
#word2state, state2word = assign_states2(num_states, states_per_word, num_words, words_per_state)
#print("assignment done")

#sys.exit()

assigned_states = functools.reduce(
    lambda acc, x: acc | set(x),
    chmm.word2state,
    set(),
)
print(len(assigned_states))

# model hacks
dim = 256

# start
start_emb = nn.Embedding(num_states, dim)
start_emb.weight = chmm.start_emb
start_mlp = chmm.start_mlp

# transition
left_emb = chmm.state_emb
trans_mlp = chmm.trans_mlp

# emission
preterminal_emb = chmm.preterminal_emb
terminal_mlp = chmm.terminal_mlp

# only get conditional distributions for allowed states
# emission: N x V
def emission(states):
    unmasked_logits = terminal_mlp(preterminal_emb(states))
    unmasked_logits[:,:,:,-1] = float("-inf")
    batch, time, k, v = unmasked_logits.shape
    logits = unmasked_logits.gather(-1, chmm.state2word[states])
    return logits.log_softmax(-1)

def full_emission():
    unmasked_logits = terminal_mlp(preterminal_emb.weight)
    unmasked_logits[:,-1] = float("-inf")
    # manually mask each emission distribution
    mask = th.zeros_like(unmasked_logits).scatter(
        -1,
        chmm.state2word,
        1,
    ).bool()
    logits = unmasked_logits.masked_fill(~mask, float("-inf"))
    return logits.log_softmax(-1)

# transition: N x K
def transition(states):
    return trans_mlp(left_emb(states)).log_softmax(-1)

def full_transition():
    return trans_mlp(left_emb.weight).log_softmax(-1)

# start: K
def start():
    return start_mlp(start_emb.weight).squeeze(-1).log_softmax(-1)

text = th.LongTensor([
    [0,1,3,4,2,5,1,5],
    [0,6,1,3,2,5,1,7],
])
batch, time = text.shape

# go from tokens to containing hidden states
sa = chmm.word2state
# states: batch x time x M
clamped_states = sa[text]

# get initial states
init = start()[clamped_states[:,0]]

# get transition matrices
# redundant computation, it's fine for now
tr = transition(clamped_states[:,:-1])
batch, timem1, k, v = tr.shape
trans_pot = tr.gather(-1, clamped_states[:, 1:, None, :].expand(batch, timem1, k, k))

# get emission distributions
em = emission(clamped_states)

# write new adaptor
log_potentials = trans_pot
# add initial state distribution
log_potentials[:,0] += init.unsqueeze(-1)
# add observations: batch x time x max_state x 1
#obs = em.gather(-1, text[:,:,None,None].expand(batch, time, M, 1))
print(clamped_states.shape)
print(chmm.state2word.shape)


ok = em[(chmm.state2word[clamped_states] == text.view(batch, time, 1, 1))]
obs = em[(chmm.state2word[clamped_states] == text.view(batch, time, 1, 1))].view(batch, time, states_per_word, 1)
log_potentials += obs[:,1:].transpose(-1, -2)
log_potentials[:,0] += obs[:,0]

ts_log_pots = log_potentials.transpose(-1, -2)


with th.no_grad():
    Z = ts.LinearChain().sum(ts_log_pots)
    #ts.LinearChain.arhmm(trans_pot.transpose(-1, -2), em.transpose(-1, -2), init, text)
    Z1 = ts.LinearChain().sum(ts.LinearChain.hmm(
        transition = full_transition().transpose(-1, -2),
        emission = full_emission().transpose(-1, -2),
        init = start(),
        observations = text,
    ))
    print(Z)
    print(Z1)
    Z2 = ts.LinearChain().sum(chmm.log_potentials(text))
    print(Z2)

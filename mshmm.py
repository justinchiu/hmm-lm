import sys
import functools

from itertools import product

import numpy as np

import torch as th
import torch_struct as ts
import torch.nn as nn

from models.mshmmlm import MshmmLm

from utils import Pack

num_states = 16
num_clusters = 4
num_repeats = num_states // num_clusters

num_words = 10

T = th.arange(num_states ** 2).view(num_states, num_states)
c2s = th.LongTensor([
    list(range(c * num_repeats, (c+1) * num_repeats))
    for c in range(num_clusters)
])
w2c = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
w2s = th.stack([
    c2s[x] for x in w2c
])

# sample mask so that we get drop p of each state
# oh. just have to gather from word2state.

# subselect half
I = th.distributions.Gumbel(0, 1).sample(c2s.shape).topk(2, dim=-1).indices
# hope not too slow...

# what's nice about brown is that the assignments are *disjoint*
# so we can turn off half the states with a gather

c2s_d = c2s.gather(1, I)

num_repeats_d = num_repeats // 2
c2sw_d = th.LongTensor([
    list(range(c * num_repeats_d, (c+1) * num_repeats_d))
    for c in range(num_clusters)
])
w2s_d = c2sw_d[w2c]

# cool, what we needed was w2s_d.
# now we need to compute the transitions and emissions sparse

H = 128
semb = th.randn(num_states, H)
temb = th.randn(num_states, H)
wemb = th.randn(num_words, H)

mask = th.zeros(num_states, dtype=th.bool)
mask[c2s_d.view(-1)] = 1
mask = ~mask

semb_f = semb.masked_fill(mask[:,None].expand(num_states, H), float("-inf"))
temb_f = temb.masked_fill(mask[:,None].expand(num_states,H), float("-inf"))

semb_d = semb[c2s_d.view(-1)]
temb_d = temb[c2s_d.view(-1)]

# transition

t_logits_d = (semb_d[:,None] * temb_d[None,:]).sum(-1).log_softmax(-1)

t_logits = (semb_f[:,None] * temb_f[None,:]).sum(-1)
t_logits[t_logits != t_logits] = float("-inf")
t_logits = t_logits.log_softmax(-1)
t_logits[t_logits != t_logits] = float("-inf")

# emission
a = (th.arange(0, num_words)[:, None]
    .expand(num_words, num_repeats // 2)
    .contiguous()
    .view(-1)
)
v = th.ones(num_words * num_repeats // 2)

i = th.stack([w2s_d.view(-1), a])
sparse = th.sparse.ByteTensor(i, v, th.Size([num_states // 2, num_words]))
emaskd = ~sparse.to_dense().bool()

e_logits_d = ((temb_d[:,None] * wemb[None,:])
    .sum(-1)
    .masked_fill(emaskd, float("-inf"))
    .log_softmax(-1)
)

# masked version
a = (th.arange(0, num_words)[:, None]
    .expand(num_words, num_repeats)
    .contiguous()
    .view(-1)
)
v = th.ones(num_words * num_repeats)

i = th.stack([w2s.view(-1), a])
sparse = th.sparse.ByteTensor(i, v, th.Size([num_states, num_words]))
emaskf = ~sparse.to_dense().bool()

e_logits = ((temb_f[:,None] * wemb[None,:])
    .sum(-1)
    .masked_fill(emaskf, float("-inf"))
    .log_softmax(-1)
)
e_logits[e_logits != e_logits] = float("-inf")

# get potentials
text = th.LongTensor([0,1,5,8,7,9])
states = w2s[text]
log_pots = t_logits[states[:-1,:,None],states[1:,None,:]]

states_d = w2s_d[text]
log_pots_d = t_logits_d[states_d[:-1,:,None], states_d[1:,None,:]]

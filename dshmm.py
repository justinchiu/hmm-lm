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
num_states = 1024
num_clusters = 64
num_repeats = num_states // num_clusters

num_words = 10
num_words = 10000

T = th.arange(num_states ** 2).view(num_states, num_states)
c2s = th.LongTensor([
    list(range(c * num_repeats, (c+1) * num_repeats))
    for c in range(num_clusters)
])
w2c = np.random.randint(num_clusters, size=(num_words,))
w2s = th.stack([
    c2s[x] for x in w2c
])

p = 0.5

dmask = th.empty(num_states).fill_(p).bernoulli_().bool()
mask = ~dmask
s2d = mask.cumsum(0) - 1

# need padding states
n = mask.sum().item()
s2d_pad = s2d.masked_fill(dmask, n)

# need to obtain s2w mask for emission

# cool, what we needed was w2s_d.
# now we need to compute the transitions and emissions sparse

H = 128
semb = th.randn(num_states, H)
temb = th.randn(num_states, H)
wemb = th.randn(num_words, H)

semb_f = semb.masked_fill(dmask[:,None].expand(num_states, H), float("-inf"))
temb_f = temb.masked_fill(dmask[:,None].expand(num_states,H), float("-inf"))

semb_d = th.cat([
    semb[mask],
    th.zeros(1, H),
], 0)
temb_d = th.cat([
    temb[mask],
    th.zeros(1, H),
], 0)

# transition

t_logits_d = (semb_d[:,None] * temb_d[None,:]).sum(-1)
t_logits_d[-1] = float("-inf")
t_logits_d[:,-1] = float("-inf")
t_logits_d = t_logits_d.log_softmax(-1)
t_logits_d[t_logits_d != t_logits_d] = float("-inf")

t_logits = (semb_f[:,None] * temb_f[None,:]).sum(-1)
t_logits[t_logits != t_logits] = float("-inf")
t_logits = t_logits.log_softmax(-1)
t_logits[t_logits != t_logits] = float("-inf")

# emission
a = (th.arange(0, num_words)[:, None]
    .expand(num_words, num_repeats)
    .contiguous()
    .view(-1)
)
v = th.ones(num_words * num_repeats)

#import pdb; pdb.set_trace()

def iw2s(w2s):
    i = th.stack([w2s.view(-1), a])
    sparse = th.sparse.ByteTensor(i, v, th.Size([n+1, num_words]))
    return ~sparse.to_dense().clamp(0, 1).bool()

print("mask")
print(mask)
print("w2s")
print(w2s)
print("s2d_pad")
print(s2d_pad)
print("s2d_pad[w2s]")
print(s2d_pad[w2s])

i = th.stack([s2d_pad[w2s].view(-1), a])
sparse = th.sparse.ByteTensor(i, v, th.Size([n+1, num_words]))
emaskd = ~sparse.to_dense().clamp(0, 1).bool()

#s2d_pad = s2d.masked_fill(dmask, -1)
w2s_d = s2d_pad[w2s]

"""
x = s2d_pad[w2s].view(-1)
m = x >= 0
i = th.stack([x[m], a[m]])
sparse = th.sparse.ByteTensor(i, v[m], th.Size([n+1, num_words]))
emaskd = ~sparse.to_dense().clamp(0, 1).bool()
"""

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

print((log_pots == log_pots_d).all())
import pdb; pdb.set_trace()

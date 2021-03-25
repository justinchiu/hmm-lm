# -*- coding: utf-8 -*-
"""kernelhmm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X5UDkWNjUn3azgvToDjQim_C2fZff0pX
"""

import math
import random

import torch
import torch_struct
import numpy as np

from assign import convert_w2s
#from genbmm import logbmm

#torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0")
seed = 1234
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

N = 3 # batch size
T = 32 # length of sequence
V = 128 # vocab size

C = 64 # number of classes
H = 32 # embedding dimension
D = 16 # number of samples / projection dim

temp = 1
states_per_word = 4
words_per_cluster = 8
num_clusters = V // words_per_cluster

start_emb = torch.randn(H, device=device) / temp
state_emb = torch.randn(C, H, device=device) / temp
next_state_emb = torch.randn(C, H, device=device) / temp
preterminal_emb = torch.randn(C, H, device=device) / temp
terminal_emb = torch.randn(V, H, device=device) / temp
#projection = torch.randn(H, D, device=device) / temp
projections = torch.randn(num_clusters, H, D, device=device) / temp
start_projection = torch.randn(H, D, device=device) / temp

start_emb.requires_grad = True
state_emb.requires_grad = True
next_state_emb.requires_grad = True
preterminal_emb.requires_grad = True
terminal_emb.requires_grad = True
projections.requires_grad = True
start_projection.requires_grad = True

# from_numpy api seems bad
text = torch.from_numpy(np.random.choice(V, size=(N, T))).to(device)
lengths = torch.from_numpy(np.random.choice(np.arange(T-3, T), size=(N,))).to(device)
lengths[0] = T
mask = torch.arange(T, device=device)[None] < lengths[:,None]
# no masking for now
print(text)
print(lengths)

# simplest mapping: divide states
word2cluster = torch.tensor([
    v // 8 for v in range(V)
])
# probably a smarter way to do this, but this is just for testing
word2state = torch.stack([
    torch.arange(
        (v // words_per_cluster) * states_per_word,
        (v // words_per_cluster) * states_per_word + states_per_word)
    for v in range(V)
])
a = (torch.arange(0, V)[:, None]
    .expand(V, states_per_word)
    .contiguous()
    .view(-1)
)
v = torch.ones(V * states_per_word)
i = torch.stack([word2state.view(-1), a])
sparse = torch.sparse.ByteTensor(i, v, torch.Size([C, V]))
mask = sparse.to_dense().bool().to(device)

state2cluster = torch.arange(num_clusters).repeat_interleave(states_per_word)

big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
#log_phi_w0 = (state_emb[:,None] @ big_projections)[:,0]
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
#log_phi_u0 = next_state_emb[:,None] @ big_projections
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[state2cluster,:]).logsumexp(-1).log_softmax(-1)
emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))
# (emission_pre[:,0].exp() > 1e-2).nonzero()
# emission[:,0].exp().nonzero()

# TORCH STRUCT
log_potentials = torch_struct.LinearChain.hmm(
    transition = transition.T,
    emission = emission.T,
    init = start,
    observations = text,
)
#evidence = torch_struct.LinearChain().sum(log_potentials, lengths=lengths)
evidence = torch_struct.LinearChain().sum(log_potentials)
print(evidence)
# TODO: check grads
evidence.sum().backward()

# clone grad then zero
start_emb_grad = start_emb.grad.detach().clone()
state_emb_grad = state_emb.grad.detach().clone()
next_state_emb_grad = next_state_emb.grad.detach().clone()
preterminal_emb_grad = preterminal_emb.grad.detach().clone()
terminal_emb_grad = terminal_emb.grad.detach().clone()
projection_grad = projections.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projections.grad.zero_()
start_projection.grad.zero_()

# LOOP
big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[state2cluster,:]).logsumexp(-1).log_softmax(-1)

emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))
# (emission_pre[:,0].exp() > 1e-2).nonzero()
# emission[:,0].exp().nonzero()
 
# gather emission
# N x T x C
p_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
alphas0 = []
#alpha = start * p_emit[:,0] # {N} x C
alpha = start + p_emit[:,0]
alphas0.append(alpha)
for t in range(T-1):
    # logbmm
    alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
    #alpha = (alpha @ transition) * p_emit[:,t+1]
    alphas0.append(alpha)
evidence_slow = alpha.logsumexp(-1)
print(evidence_slow)

# LOOPBMM
big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[state2cluster,:]).logsumexp(-1).softmax(-1)

emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))
# (emission_pre[:,0].exp() > 1e-2).nonzero()
# emission[:,0].exp().nonzero()
 
# gather emission
# N x T x C
p_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
alphas_bmm = []
evidences_bmm = []
alpha_un = start + p_emit[:,0] # {N} x C
Ot = alpha_un.logsumexp(-1, keepdim=True)
alpha = (alpha_un - Ot).exp()
alphas_bmm.append(alpha)
evidences_bmm.append(Ot)
for t in range(T-1):
    # logbmm
    #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
    alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()
    alphas_bmm.append(alpha)
    evidences_bmm.append(Ot)
O = torch.cat(evidences_bmm, -1)
evidence_slow_bmm = O.sum(-1)
print("LOOPBMM")
print(evidence_slow_bmm)

# SPARSELOOPBMM
big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[state2cluster,:]).logsumexp(-1).softmax(-1)

emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))
# (emission_pre[:,0].exp() > 1e-2).nonzero()
# emission[:,0].exp().nonzero()
 
# gather emission
# N x T x C
p_emit0 = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
state_t = word2state[text]
p_emit = emission[
    state_t,
    text[:,:,None],
]

transitions = transition[state_t[:,:-1,:,None], state_t[:,1:,None,:]]

alphas_bmm = []
evidences_bmm = []

alpha_un0 = start + p_emit0[:,0] 
Ot0 = alpha_un0.logsumexp(-1, keepdim=True)
alpha0 = (alpha_un0 - Ot0).exp()

alpha_un = start[state_t[:,0]] + p_emit[:,0] # {N} x S
Ot = alpha_un.logsumexp(-1, keepdim=True)
alpha = (alpha_un - Ot).exp()
alphas_bmm.append(alpha)
evidences_bmm.append(Ot)
for t in range(T-1):
    # logbmm
    #alpha = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)
    alpha_un0 = (alpha0 @ transition).log() + p_emit0[:,t+1]
    Ot0 = alpha_un0.logsumexp(-1, keepdim=True)
    alpha0 = (alpha_un0 - Ot0).exp()

    # hm, not sure if this is slow though?
    alpha_un = (alpha[:,None] @ transitions[:,t])[:,0].log() + p_emit[:,t+1]
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()

    alphas_bmm.append(alpha)
    evidences_bmm.append(Ot)
O = torch.cat(evidences_bmm, -1)
evidence_slow_bmm = O.sum(-1)
print("SPARSELOOPBMM")
print(evidence_slow_bmm)

# LOOP_FAST_BMM
big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
#transition = (log_phi_w @ log_phi_u.T).softmax(-1)
emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))

# O(CD)
log_denominator = (
    log_phi_w.view(num_clusters, states_per_word, D)
    + log_phi_u.logsumexp(1, keepdim=True)
).logsumexp(-1).view(C)
# O(CD)
normed_log_phi_w = log_phi_w - log_denominator.unsqueeze(-1)

normalized_phi_w = normed_log_phi_w.exp()
phi_u = log_phi_u.exp()

# gather emission
# N x T x C
p_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
alphas = []
Os = []
#alpha = start * p_emit[:,0] # {N} x C
alpha_un = start + p_emit[:,0]
Ot = alpha_un.logsumexp(-1, keepdim=True)
alpha = (alpha_un - Ot).exp()
alphas.append(alpha)
Os.append(Ot)
for t in range(T-1):
    # N x C x D
    gamma = alpha.unsqueeze(-1) * normalized_phi_w
    alpha_un = p_emit[:,t+1] + torch.einsum(
        "ncsd,czd->nz",
        gamma.view(N, num_clusters, states_per_word, D),
        phi_u,
    ).log()
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()

    alphas.append(alpha)
    Os.append(Ot)
O = torch.cat(Os, -1)
evidence_fast_bmm = O.sum(-1)
print("FASTBMM")
print(evidence_fast_bmm)

# SPARSE_LOOP_FAST_BMM
big_projections = projections[state2cluster]
log_phi_w_start = start_emb @ start_projection
log_phi_u_start = next_state_emb @ start_projection
log_phi_w = torch.einsum("sd,sdf->sf", state_emb, big_projections)
log_phi_u = torch.einsum("sd,cdf->csf", next_state_emb, projections)

start = (log_phi_w_start + log_phi_u_start).logsumexp(-1).log_softmax(-1)
#transition = (log_phi_w @ log_phi_u.T).softmax(-1)
emission_logits = (preterminal_emb @ terminal_emb.T)
emission_logits.masked_fill_(~mask, -1e7)
#emission = emission_logits.log_softmax(-1)
# issues when some states dont emit any words, but doesnt matter for now since computing evidence
emission_pre = emission_logits.log_softmax(-1)
emission = emission_pre.masked_fill(~mask, float("-inf"))

# O(CD)
log_denominator = (
    log_phi_w.view(num_clusters, states_per_word, D)
    + log_phi_u.logsumexp(1, keepdim=True)
).logsumexp(-1).view(C)
# O(CD)
normed_log_phi_w = log_phi_w - log_denominator.unsqueeze(-1)

normalized_phi_w = normed_log_phi_w.exp()
phi_u = log_phi_u.exp()

# gather emission
# N x T x C
p_emit0 = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]

state_t = word2state[text]
cluster_t = word2cluster[text]
left_proj = normalized_phi_w[state_t[:,:-1]]
right_proj = phi_u[cluster_t[:,:-1,None],state_t[:,1:]]

p_emit = emission[
    state_t,
    text[:,:,None],
]

alphas = []
Os = []

alpha_un = start[state_t[:,0]] + p_emit[:,0]
Ot = alpha_un.logsumexp(-1, keepdim=True)
alpha = (alpha_un - Ot).exp()
alphas.append(alpha)
Os.append(Ot)
for t in range(T-1):
    gamma = torch.einsum("ns,nsd->nd", alpha, left_proj[:,t])
    alpha_un = p_emit[:,t+1] + torch.einsum(
        "nd,nsd->ns", gamma, right_proj[:,t],
    ).log()
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()

    alphas.append(alpha)
    Os.append(Ot)
O = torch.cat(Os, -1)
evidence_fast_bmm = O.sum(-1)
print("SPARSE LOOP FAST BMM")
print(evidence_fast_bmm)


import pdb; pdb.set_trace()
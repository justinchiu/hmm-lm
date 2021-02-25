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

C = 256 # number of classes
H = 128 # embedding dimension
D = 64 # number of samples / projection dim

start_emb = torch.randn(H, device=device)
state_emb = torch.randn(C, H, device=device)
next_state_emb = torch.randn(C, H, device=device)
preterminal_emb = torch.randn(C, H, device=device)
terminal_emb = torch.randn(V, H, device=device)
projection = torch.randn(H, D, device=device)

start_emb.requires_grad = True
state_emb.requires_grad = True
next_state_emb.requires_grad = True
preterminal_emb.requires_grad = True
terminal_emb.requires_grad = True
projection.requires_grad = True

# from_numpy api seems bad
text = torch.from_numpy(np.random.choice(V, size=(N, T))).to(device)
lengths = torch.from_numpy(np.random.choice(np.arange(T-3, T), size=(N,))).to(device)
lengths[0] = T
mask = torch.arange(T, device=device)[None] < lengths[:,None]
# no masking for now
print(text)
print(lengths)

log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)

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
projection_grad = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

# LOOP
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[None]).logsumexp(-1).log_softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
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
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[None]).logsumexp(-1).softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
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
print(evidence_slow_bmm)

# LOOP_FAST
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
#transition = (log_phi_w @ log_phi_u.T).softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
# O(CD)
log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
# O(CD)
normed_log_phi_w = log_phi_w - log_denominator[:,None]
# gather emission
# N x T x C
p_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
alphas = []
#alpha = start * p_emit[:,0] # {N} x C
alpha = start + p_emit[:,0]
alphas.append(alpha)
for t in range(T-1):
    alpha_slow = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)

    # for a single timestep, we project previous alpha, ie posterior over last state
    # given words up to t, compute next alpha by projection to feature space and back

    # logmm: (N,C) @ (C,D)
    # N = batch
    # C = num states
    # D = num features
    logmm = lambda x,y: (x[:,:,None] + y[None]).logsumexp(1)
    beta0 = logmm(alpha, normed_log_phi_w)
    alpha0 = p_emit[:,t+1] + logmm(beta0, log_phi_u.T)

    beta = (alpha[:,:,None] + log_phi_w[None] - log_denominator[None,:,None]).logsumexp(-2)
    alpha = p_emit[:,t+1] + (log_phi_u[None] + beta[:,None]).logsumexp(-1)

    # logbmm
    #alpha = (alpha @ transition) * p_emit[:,t+1]
    alphas.append(alpha)
evidence_fast = alpha.logsumexp(-1)
print(evidence_fast)


# LOOP_FAST_BMM
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
#transition = (log_phi_w @ log_phi_u.T).softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
# O(CD)
log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
# O(CD)
normed_log_phi_w = log_phi_w - log_denominator[:,None]

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
    gamma = alpha @ normalized_phi_w
    alpha_un = p_emit[:,t+1] + (gamma @ phi_u.T).log()
    Ot = alpha_un.logsumexp(-1, keepdim=True)
    alpha = (alpha_un - Ot).exp()

    alphas.append(alpha)
    Os.append(Ot)
O = torch.cat(Os, -1)
evidence_fast_bmm = O.sum(-1)
print(evidence_fast_bmm)


import pdb; pdb.set_trace()

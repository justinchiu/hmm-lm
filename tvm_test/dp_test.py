# -*- coding: utf-8 -*-

import math
import random

import torch
import torch_struct
import numpy as np

device = torch.device("cuda:0")
#device = torch.device("cpu")
seed = 1234
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

N = 3 # batch size
T = 8 # length of sequence
V = 128 # vocab size

C = 64 # number of classes
H = 128 # embedding dimension
D = 32 # number of samples / projection dim

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

# from_numpy api does not take device
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
crf = torch_struct.LinearChainCRF(log_potentials)
evidence = crf.partition
edge_marginals = crf.marginals
# check grads
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
# check grads
evidence_slow.sum().backward()

# clone grad then zero
start_emb_grad_loop = start_emb.grad.detach().clone()
state_emb_grad_loop = state_emb.grad.detach().clone()
next_state_emb_grad_loop = next_state_emb.grad.detach().clone()
preterminal_emb_grad_loop = preterminal_emb.grad.detach().clone()
terminal_emb_grad_loop = terminal_emb.grad.detach().clone()
projection_grad_loop = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

# LOOP_FAST
# 1xD
log_phi_start = start_emb @ projection
# CxD
log_phi_w = state_emb @ projection
# CxD
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

    # alpha: N x C
    # log_phi_w: C x D {normed_log_phi_w as well}
    # beta: N x D
    #beta = (alpha[:,:,None] + log_phi_w[None] - log_denominator[None,:,None]).logsumexp(-2)
    beta = (alpha[:,:,None] + normed_log_phi_w[None]).logsumexp(-2)
    # p_emit[:,t+1]: N x C 
    alpha = p_emit[:,t+1] + (log_phi_u[None] + beta[:,None]).logsumexp(-1)

    # logbmm
    #alpha = (alpha @ transition) * p_emit[:,t+1]
    alphas.append(alpha)
evidence_fast = alpha.logsumexp(-1)
# check grads
evidence_fast.sum().backward()

# clone grad then zero
start_emb_grad_loopfast = start_emb.grad.detach().clone()
state_emb_grad_loopfast = state_emb.grad.detach().clone()
next_state_emb_grad_loopfast = next_state_emb.grad.detach().clone()
preterminal_emb_grad_loopfast = preterminal_emb.grad.detach().clone()
terminal_emb_grad_loopfast = terminal_emb.grad.detach().clone()
projection_grad_loopfast = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

# LOOP_Manual
# 1xD
log_phi_start = start_emb @ projection
# CxD
log_phi_w = state_emb @ projection
# CxD
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

# implement in TVM

# logmm: (N,C) @ (C,D)
# N = batch
# C = num states
# D = num features
logmm = lambda x,y: (x[:,:,None] + y[None,:,:]).logsumexp(1)

alphas = []
gammas = []
alpha = start + p_emit[:,0]
alphas.append(alpha)
for t in range(T-1):
    #alpha_slow = (alpha[:,:,None] + transition[None] + p_emit[:,t+1,None,:]).logsumexp(-2)

    # for a single timestep, we project previous alpha, ie posterior over last state
    # given words up to t, compute next alpha by projection to feature space and back

    # beta: N x D
    gamma = logmm(alpha, normed_log_phi_w)
    # alpha: N x C
    alpha = p_emit[:,t+1] + logmm(gamma, log_phi_u.T)

    alphas.append(alpha)
    gammas.append(gamma)
evidence_manual = alpha.logsumexp(-1)

betas = []
xis = []
# backward
beta = torch.zeros(N, C, device=device)#.fill_(math.log(1/C))
betas.append(beta)
for t in range(T-1,0,-1):
    # sanity check, beta_slow == beta
    #transition = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
    #beta_slow = logmm(p_emit[:,t] + beta, transition.T)

    xi = logmm(p_emit[:,t] + beta, log_phi_u)
    beta = logmm(xi, normed_log_phi_w.T)

    betas.append(beta)
    xis.append(xi)
last_beta = beta + p_emit[:,0] + start
# dont add last beta, not needed. can obtain from alpha[:,0] + beta[:,0]
#betas.append(last_beta)
aligned_betas = list(reversed(betas))
aligned_xis = list(reversed(xis))
alpha = torch.stack(alphas, 1)
gamma = torch.stack(gammas, 1)
beta = torch.stack(aligned_betas, 1)
xi = torch.stack(aligned_xis, 1)

# / implement in TVM

# sanity checks
log_marginals = alpha + beta
normed_log_marginals = log_marginals.log_softmax(-1)
ts_marginals = torch.cat(
    (edge_marginals[:,0].sum(-2, keepdim=True), edge_marginals.sum(-1)),
    dim = 1,
)
print("marginal diff", (ts_marginals - normed_log_marginals.exp()).abs().max().item())
print("marginal diff manual norm", (ts_marginals - (log_marginals - evidence_manual[:,None,None]).exp()).abs().max().item())
marg_diff = normed_log_marginals[:,:-1] - (
    alpha[:,:-1,:,None] + normed_log_phi_w + xi[:,:,None,:] - evidence[:,None,None,None]
).logsumexp(-1)
print("log marg diff alph w xi", marg_diff.abs().max().item())
marg_diff = normed_log_marginals[:,1:] - (
    gamma[:,:,None,:]
    + p_emit[:,1:,:,None] + log_phi_u
    #+ beta[:,:-1,:,None] # drop the first beta term since gamma has one extra operation vs alpha[:,0]
    + beta[:,1:,:,None]
    - evidence[:,None,None,None]
).logsumexp(-1)
print("log marg diff gamma u beta", marg_diff.abs().max().item())
# / sanity checks

# gradient wrt p_emit, log_phi_u, and log_phi_w
emit_loss = p_emit * log_marginals.softmax(-1).detach()
start_loss = start * log_marginals[:,0].softmax(-1).detach()
log_phi_w_loss = normed_log_phi_w * (
    alpha[:,:-1,:,None] + normed_log_phi_w + xi[:,:,None,:] - evidence_manual[:,None,None,None]
).view(-1, C, D).logsumexp(0).exp().detach()
log_phi_u_loss = log_phi_u * (
    gamma[:,:,None,:] + p_emit[:,1:,:,None] + log_phi_u + beta[:,1:,:,None] - evidence_manual[:,None,None,None]
).view(-1, C, D).logsumexp(0).exp().detach()
elbo_manual = emit_loss.sum() + start_loss.sum() + log_phi_w_loss.sum() + log_phi_u_loss.sum()
# check grads
elbo_manual.backward()

# clone grad then zero
start_emb_grad_manual = start_emb.grad.detach().clone()
state_emb_grad_manual = state_emb.grad.detach().clone()
next_state_emb_grad_manual = next_state_emb.grad.detach().clone()
preterminal_emb_grad_manual = preterminal_emb.grad.detach().clone()
terminal_emb_grad_manual = terminal_emb.grad.detach().clone()
projection_grad_manual = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

# grad check
pairs = [
    (start_emb_grad_manual, start_emb_grad,),
    (preterminal_emb_grad_manual, preterminal_emb_grad,),
    (terminal_emb_grad_manual, terminal_emb_grad,),
    (state_emb_grad_manual, state_emb_grad,),
    (next_state_emb_grad_manual, next_state_emb_grad,),
    (projection_grad_manual, projection_grad,),
]
for i, (x, y) in enumerate(pairs):
    grad_diff = (x-y).abs().max()
    print(i, grad_diff.item())
import pdb; pdb.set_trace()

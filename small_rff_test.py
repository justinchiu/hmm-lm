# -*- coding: utf-8 -*-
"""kernelhmm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X5UDkWNjUn3azgvToDjQim_C2fZff0pX
"""

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "hmm_runners/hmm.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import math
import random

import torch
import torch_struct
import numpy as np
from genbmm import logbmm

torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
D = 128 # number of samples / projection dim

start_emb = torch.randn(H)
state_emb = torch.randn(C, H)
next_state_emb = torch.randn(C, H)
preterminal_emb = torch.randn(C, H)
terminal_emb = torch.randn(V, H)
projection = torch.randn(H, D)

start_emb.requires_grad = True
state_emb.requires_grad = True
next_state_emb.requires_grad = True
preterminal_emb.requires_grad = True
terminal_emb.requires_grad = True
projection.requires_grad = True

text = torch.from_numpy(np.random.choice(V, size=(N, T))).cuda()
lengths = torch.from_numpy(np.random.choice(np.arange(T-3, T), size=(N,))).cuda()
lengths[0] = T
mask = torch.arange(T)[None] < lengths[:,None]
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
evidence = torch_struct.LinearChain().sum(log_potentials, lengths=lengths)
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

# RFF LOG-SPACE VERSION
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
logp_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

# EFFICIENCY: anything involving C we want to checkpoint away / logbmm / tvm / custom bwd

## First term
# D
log_sum_phi_u = log_phi_u.logsumexp(0)
# SCALAR, logdot
log_start_denominator = (log_phi_start + log_sum_phi_u).logsumexp(0)
# D
log_start_vec = log_phi_start - log_start_denominator

## Middle terms
# C = C x D + D
log_denominator = (log_phi_w + log_sum_phi_u).logsumexp(-1)
# C x Du x {Dw} + C x {Du} x Dw
log_numerator = log_phi_u[:,:,None] + log_phi_w[:,None,:]
# C x Du x Dw
log_trans_mat = log_numerator - log_denominator[:,None,None]

"""correct masking"""
log_potentials0 = logbmm(
    logp_emit.view(1, -1, C), # N x T x C
    log_trans_mat.view(1, C, -1), # C x D x D
).view(N, T, D, D)
log_eye = torch.empty(D,D).fill_(float("-inf"))
log_eye.diagonal().fill_(0.)
log_potentials0 = torch.where(mask[:,:,None,None], log_potentials0, log_eye[None,None])

log_end_vec0 = logbmm(
    logp_emit[None,torch.arange(N),lengths-1],
    log_phi_u[None],
)

log_potentials0[torch.arange(N), lengths-1,:,0] = log_end_vec0
log_potentials0[torch.arange(N), lengths-1,:,1:] = float("-inf")
log_potentials0[:,0] += log_start_vec[None,:,None]
evidence0 = torch_struct.LinearChain().sum(log_potentials0.transpose(-1, -2))
evidence0.sum().backward()

start_emb_grad0 = start_emb.grad.detach().clone()
state_emb_grad0 = state_emb.grad.detach().clone()
next_state_emb_grad0 = next_state_emb.grad.detach().clone()
preterminal_emb_grad0 = preterminal_emb.grad.detach().clone()
terminal_emb_grad0 = terminal_emb.grad.detach().clone()
projection_grad0 = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

# softmax with tvm
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
transition = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)

log_potentials = torch_struct.LinearChain.hmm(
    transition = transition.T,
    emission = emission.T,
    init = start,
    observations = text,
)
fb = foo.get_fb(C)
with torch.no_grad():
    log_m, alphas = fb(log_potentials.detach().clone().to(dtype=torch.float32).cuda(), mask=mask)
idx = torch.arange(N)
alpha_T2 = alphas[lengths-1, idx]
evidence2 = alpha_T2.logsumexp(-1)
elbo2 = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()
elbo2.sum().backward()

# clone grad then zero
start_emb_grad2 = start_emb.grad.detach().clone()
state_emb_grad2 = state_emb.grad.detach().clone()
next_state_emb_grad2 = next_state_emb.grad.detach().clone()
preterminal_emb_grad2 = preterminal_emb.grad.detach().clone()
terminal_emb_grad2 = terminal_emb.grad.detach().clone()
projection_grad2 = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()


# another rff one with tvm
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)
logp_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

# EFFICIENCY: anything involving C we want to checkpoint away / logbmm / tvm / custom bwd

## First term
# D
log_sum_phi_u = log_phi_u.logsumexp(0)
# SCALAR, logdot
log_start_denominator = (log_phi_start + log_sum_phi_u).logsumexp(0)
# D
log_start_vec = log_phi_start - log_start_denominator

## Middle terms
# C = C x D + D
log_denominator = (log_phi_w + log_sum_phi_u).logsumexp(-1)
# C x Du x {Dw} + C x {Du} x Dw
log_numerator = log_phi_u[:,:,None] + log_phi_w[:,None,:]
# C x Du x Dw
log_trans_mat = log_numerator - log_denominator[:,None,None]

"""correct masking"""
log_potentials0 = logbmm(
    logp_emit.view(1, -1, C), # N x T x C
    log_trans_mat.view(1, C, -1), # C x D x D
).view(N, T, D, D)
log_eye = torch.empty(D,D).fill_(float("-inf"))
log_eye.diagonal().fill_(0.)
log_potentials0 = torch.where(mask[:,:,None,None], log_potentials0, log_eye[None,None])

log_end_vec0 = logbmm(
    logp_emit[None,torch.arange(N),lengths-1],
    log_phi_u[None],
)

log_potentials0[torch.arange(N), lengths-1,:,0] = log_end_vec0
log_potentials0[torch.arange(N), lengths-1,:,1:] = float("-inf")
log_potentials0[:,0] += log_start_vec[None,:,None]
log_potentials0 = log_potentials0.transpose(-1, -2)

fbd = foo.get_fb(D)
with torch.no_grad():
    log_m, alphas = fbd(
        log_potentials0.detach().clone().to(dtype=torch.float32).cuda(),
        mask=torch.cat([mask[:,(0,)],mask], dim=-1),
    )
idx = torch.arange(N)
alpha_T = alphas[lengths, idx]
evidence1 = alpha_T.logsumexp(-1)
log_potentials0[log_potentials0 == float("-inf")] = 0
elbo1 = (log_m.exp() * log_potentials0)[mask].sum()
elbo1.backward()

start_emb_grad1 = start_emb.grad.detach().clone()
state_emb_grad1 = state_emb.grad.detach().clone()
next_state_emb_grad1 = next_state_emb.grad.detach().clone()
preterminal_emb_grad1 = preterminal_emb.grad.detach().clone()
terminal_emb_grad1 = terminal_emb.grad.detach().clone()
projection_grad1 = projection.grad.detach().clone()

start_emb.grad.zero_()
state_emb.grad.zero_()
next_state_emb.grad.zero_()
preterminal_emb.grad.zero_()
terminal_emb.grad.zero_()
projection.grad.zero_()

print("evidence")
print(evidence)
print(evidence0)
print(evidence2)
print(evidence1)

print("elbo")
print(elbo2)
print(elbo1)


print("grad diff")
grad_pairs = [
    (start_emb_grad,start_emb_grad0,),
    (state_emb_grad,state_emb_grad0,),
    (next_state_emb_grad,next_state_emb_grad0,),
    (preterminal_emb_grad,preterminal_emb_grad0,),
    (terminal_emb_grad,terminal_emb_grad0,),
    (projection_grad,projection_grad0,),
]
for x,y in grad_pairs:
    print((x-y).abs().max())
    print(x.max())
    print(x.min())
    print(torch.allclose(x,y, rtol=1e-3))

print("TVM")
grad_pairs = [
    (start_emb_grad,start_emb_grad1,),
    (state_emb_grad,state_emb_grad1,),
    (next_state_emb_grad,next_state_emb_grad1,),
    (preterminal_emb_grad,preterminal_emb_grad1,),
    (terminal_emb_grad,terminal_emb_grad1,),
    (projection_grad,projection_grad1,),
]
for x,y in grad_pairs:
    print((x-y).abs().max())
    print(x.max())
    print(x.min())
    print(torch.allclose(x,y, rtol=1e-3))


print("TVM2")
grad_pairs = [
    (start_emb_grad2,start_emb_grad1,),
    (state_emb_grad2,state_emb_grad1,),
    (next_state_emb_grad2,next_state_emb_grad1,),
    (preterminal_emb_grad2,preterminal_emb_grad1,),
    (terminal_emb_grad2,terminal_emb_grad1,),
    (projection_grad2,projection_grad1,),
]
for x,y in grad_pairs:
    print((x-y).abs().max())
    print(x.max())
    print(x.min())
    print(torch.allclose(x,y, rtol=1e-3))




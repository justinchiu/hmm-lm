# -*- coding: utf-8 -*-
"""kernelhmm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X5UDkWNjUn3azgvToDjQim_C2fZff0pX
"""

import math

import torch
import torch_struct
import numpy as np
from genbmm import logbmm

#torch.set_default_tensor_type(torch.cuda.FloatTensor)

def time_f(f):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    f()
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    return start.elapsed_time(end)
    # MILLISECONDS

N = 16 # batch size
T = 32 # length of sequence
V = 10000 # vocab size

C = 64 # number of classes
H = 16 # embedding dimension
D = 5 # number of samples / projection dim

start_emb = torch.randn(H)
state_emb = torch.randn(C, H)
next_state_emb = torch.randn(C, H)
preterminal_emb = torch.randn(C, H)
terminal_emb = torch.randn(V, H)

state_emb.requires_grad = True
next_state_emb.requires_grad = True

projection = torch.randn(H, D)
text = torch.from_numpy(np.random.choice(V, size=(N, T)))
# no masking for now
#print(text)

def get_times(C, H, D):
    start_emb = torch.randn(H)
    state_emb = torch.randn(C, H)
    next_state_emb = torch.randn(C, H)
    preterminal_emb = torch.randn(C, H)
    terminal_emb = torch.randn(V, H)

    state_emb.requires_grad = True
    next_state_emb.requires_grad = True

    projection = torch.randn(H, D)
    text = torch.from_numpy(np.random.choice(V, size=(N, T)))

    def slow(start_emb, state_emb, next_state_emb, preterminal_emb, terminal_emb, projection):
        # LOOP VERSION
        log_phi_start = start_emb @ projection
        log_phi_w = state_emb @ projection
        log_phi_u = next_state_emb @ projection

        start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).softmax(-1)
        transition = (log_phi_w @ log_phi_u.T).softmax(-1)
        emission = (preterminal_emb @ terminal_emb.T).softmax(-1)
        # gather emission
        # N x T x C
        p_emit = emission[
            torch.arange(C)[None,None],
            text[:,:,None],
        ]
        #log_pots = p_emit[:,1:,None,:] * transition[None,None]
        alphas = []
        alpha = start * p_emit[:,0] # {N} x C
        for t in range(T-1):
            # logbmm
            #alpha = (alpha[:,:,None] + log_pots[:,t]).logsumexp(-2)
            alpha = (alpha @ transition) * p_emit[:,t+1]
            #alpha = (alpha[:,None] @ log_pots[:,t])[:,0]
        evidence_slow = alpha.logsumexp(-1)


    def fast(start_emb, state_emb, next_state_emb, preterminal_emb, terminal_emb, projection):
        # EMBEDDED VERSION
        log_phi_start = start_emb @ projection
        log_phi_w = state_emb @ projection
        log_phi_u = next_state_emb @ projection
        phi_start = log_phi_start.exp()
        phi_w = log_phi_w.exp()
        phi_u = log_phi_u.exp()

        start = (log_phi_start[None,None] + log_phi_u[None,:]).logsumexp(-1).softmax(-1)
        emission = (preterminal_emb @ terminal_emb.T).softmax(-1)
        # gather emission
        # N x T x C
        p_emit = emission[
            torch.arange(C)[None,None],
            text[:,:,None],
        ]

        #denominator = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).logsumexp(-1).exp()
        #denominator = (phi_w @ phi_u.T).sum(-1)
        denominator = (phi_w * phi_u.sum(0, keepdim=True)).sum(-1)
        normalized_phi_w = phi_w / denominator[:,None]

        alphas_fast = []
        alpha = start * p_emit[:,0]
        alphas_fast.append(alpha)
        for t in range(T-1):
            # matvec over classes
            #beta = (alpha[:,:,None] + log_phi_w[None]).logsumexp(1)
            beta = alpha @ normalized_phi_w
            #alpha =  logp_emit[:,t+1] - log_denominator + (log_phi_u[None] + beta[:,None,]).logsumexp(-1)
            alpha = p_emit[:,t+1] + (beta @ phi_u.T)
            # logbmm
        evidence_slow = alpha.logsumexp(-1)

    slow_times = []
    for _ in range(15):
        slow_times.append(time_f(
            lambda: slow(start_emb, state_emb, next_state_emb, preterminal_emb, terminal_emb, projection)
        ))
    fast_times = []
    for _ in range(15):
        fast_times.append(time_f(
            lambda: fast(start_emb, state_emb, next_state_emb, preterminal_emb, terminal_emb, projection)
        ))

    return np.mean(slow_times[1:]), np.mean(fast_times[1:])


H = 256 # embedding dim
#D = 1 # num features
D = 512
for C in [512, 1024, 2048, 4098, 8000, 16000]: # class size
    slow_time, fast_time = get_times(C, H, D)
    print(f"Num classes: {C} | slow: {slow_time} fast: {fast_time}")

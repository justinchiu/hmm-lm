import torch
import torch_struct

from genbmm import logbmm

from torch.utils.checkpoint import checkpoint

import numpy as np

N = 2
T = 4
V = 128

C = 64
H = 16
D = 32

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
print(text)

# TODO: copy paste this for grad checking
log_phi_start = start_emb @ projection
log_phi_w = state_emb @ projection
log_phi_u = next_state_emb @ projection

start = (log_phi_start @ log_phi_u.T).log_softmax(-1)
transition = (log_phi_w @ log_phi_u.T).log_softmax(-1)
emission = (preterminal_emb @ terminal_emb.T).log_softmax(-1)

# TORCH STRUCT
log_potentials = torch_struct.LinearChain.hmm(
    transition = transition.T,
    emission = emission.T,
    init = start,
    observations = text,
)
evidence = torch_struct.LinearChain().sum(log_potentials)
# TODO: check grads


# LOOP VERSION
# gather emission
# N x T x C
logp_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]
log_pots = logp_emit[:,1:,None,:] + transition[None,None]
alpha = start[None] + logp_emit[:,0] # {N} x C
for t in range(T-1):
    # logbmm
    alpha = (alpha[:,:,None] + log_pots[:,t]).logsumexp(-2)
evidence_slow = alpha.logsumexp(-1)

# TODO: check grads


# RFF VERSION
logp_emit = emission[
    torch.arange(C)[None,None],
    text[:,:,None],
]

# EFFICIENCY: anything involving C we want to checkpoint away / logbmm / tvm / custom bwd

## First term
# D
log_sum_phi_u = log_phi_u.logsumexp(0)
# SCALAR, logdot
log_start_denominator = (log_phi_start + log_sum_phi_u).logsumexp(0)
# D
log_start_vec = log_phi_start - log_start_denominator

## Middle terms
# C
log_denominator = (log_phi_w + log_sum_phi_u).logsumexp(-1)
# C x Du x Dw
log_numerator = log_phi_u[:,:,None] + log_phi_w[:,None,:]
# C x Du x Dw
log_trans_mat = log_numerator - log_denominator[:,None,None]
# N x T x C x {Du} x {Dw} + {N} x {T} x C x Du x Dw
log_potentials = (logp_emit[:,:-1,:,None,None] + log_trans_mat[None,None]).logsumexp(2)

## Last term
# N x D
log_end_vec = (
    logp_emit[:,-1,:,None] # N x (T=-1) x C x {D}
    + log_phi_u[None] # {N} x C x D
).logsumexp(1)

# can use tvm here
evidence0 = log_start_vec[None] # {N} x Dw
for t in range(T-1):
    # logbmm
    evidence0 = (evidence0[:,:,None] + log_potentials[:,t]).logsumexp(-2)
evidence0 = (evidence0 + log_end_vec).logsumexp(-1)

print(evidence)
print(evidence_slow)
print(evidence0)

import pdb; pdb.set_trace()

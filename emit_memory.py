

import numpy as np

import torch
import torchtext

from pytorch_memlab import profile, MemReporter

from datasets.ptb import PennTreebank, BucketIterator
from assign import read_lm_clusters, assign_states_brown_cluster

device = torch.device("cuda:0")

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = PennTreebank.splits(
    TEXT,
    newline_eos = True,
)

TEXT.build_vocab(train)
V = TEXT.vocab

num_clusters = 128
num_clusters = 128
C = 2 ** 14
spw = C // num_clusters
word2cluster, word_counts, cluster2word = read_lm_clusters(
    V, path=f"clusters/lm-{num_clusters}/paths",
)
word2state, cluster2state, word2cluster, c2sw_d = assign_states_brown_cluster(
    C,
    word2cluster,
    V,
    spw,
)

word2state = torch.from_numpy(word2state).to(device)

words = 512
time = 32
batch = words // time


# backward tests
def f(x):
    from pytorch_memlab import MemReporter
    MemReporter().report()
    import pdb; pdb.set_trace()

#@profile
def h():
    x = torch.randn(2 ** 26, device=device)
    x.requires_grad = True
    #x.register_hook(f)
    #maskx = torch.cuda.LongTensor([0,1,2,3,4,3,2,1,2,3,2,1] * 50)
    maskx = torch.empty_like(x).bernoulli_(0.5).nonzero()
    print(maskx.dtype)
    print(maskx.nelement() * 8 / 2 ** 30)
    y = x[maskx[:-100]]
    a = y.sum()
    a.backward(retain_graph=True)
#h()

#@profile
def meh():
    x = torch.randn(2 ** 26, device=device)
    maskx = torch.empty_like(x).bernoulli_(0.5).nonzero()
    grad = x.index_put((maskx,), torch.ones(maskx.shape, device=device), accumulate=True)
#meh()

print(torch.cuda.max_memory_allocated() / 2 ** 30)
#@profile
def g():
    x = torch.randn(2 ** 26, device=device)
    x.requires_grad = True
    #x.register_hook(f)
    maskx = torch.empty_like(x).bernoulli_(0.5).nonzero()
    y = x[maskx]
    a = y.sum()
    """
    #y.register_hook(f)
    masky = torch.empty_like(y).bernoulli_(0.5).bool()
    z = y[masky]
    #z.register_hook(f)
    a = z.sum()
    """
    #a.register_hook(f)
    #MemReporter().report()
    a.backward(retain_graph=True)
    #MemReporter().report()
#g()
print(torch.cuda.max_memory_allocated() / 2 ** 30)

#@profile
def emit_old(C, V, spw, word2state, device):
    logits = torch.randn(C, len(V), device=device)
    logits.requires_grad = True
    logits = logits + 1
    a = (torch.arange(0, len(V), device=device)[:, None]
        .expand(len(V), spw)
        .contiguous()
        .view(-1)
    )
    v = torch.ones((len(V)) * spw, device=device)

    i = torch.stack([word2state.view(-1), a])
    C = logits.shape[0]
    sparse = torch.sparse.ByteTensor(i, v, torch.Size([C, len(V)]))
    mask = sparse.to_dense()
    mask = mask.bool()
    mask = mask.to(logits.device)
    #if wandb.run.mode == "dryrun":
        #import pdb; pdb.set_trace()
    #log_probs = logits.masked_fill(~mask, float("-inf")).log_softmax(-1)
    log_probs = logits.masked_fill_(~mask, float("-inf")).log_softmax(-1)
    loss = log_probs[mask].sum()
    loss.backward()

    #print(logits.nelement() * 4 / 2 ** 30)
    #print(mask.nelement() * 4 / 2 ** 30)

#emit_old(C, V, spw, word2state, device)

@profile
def transition_indexing(device, word2state):
    C = 2 ** 14
    spw = 512
    V = 10000


    start = torch.randn(C, device=device)
    start.requires_grad = True
    transition = torch.randn(C, C, device=device)
    transition.requires_grad = True
    emission = torch.randn(C, V, device=device)
    emission.requires_grad = True

    batch, time = 16, 32
    timem1 = time - 1

    text = torch.from_numpy(np.random.choice(
        V, size=(16,32))
    ).to(device).long()
    clamped_states = word2state[text]

    log_potentials = transition[
        clamped_states[:,:-1,:,None],
        clamped_states[:,1:,None,:],
    ]
    init = start[clamped_states[:,0]]
    obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
    log_potentials[:,0] += init.unsqueeze(-1)
    log_potentials += obs[:,1:].transpose(-1, -2)
    log_potentials[:,0] += obs[:,0]
    #if wandb.run.mode == "dryrun":
        #print(f"total clamp time: {timep.time() - start_clamp}")
    loss = log_potentials.sum()
    loss.backward()

transition_indexing(device, word2state)

class Index2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idx1, idx2):
        ctx.save_for_backward(input, idx1, idx2)
        return input[idx1, idx2]

    @staticmethod
    def backward(ctx, grad_output):
        input, idx1, idx2 = ctx.saved_tensors
        input.zero_()
        return input.index_put_((idx1, idx2), grad_output, accumulate=True), None, None

class Index1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, idx):
        ctx.save_for_backward(input, idx)
        return input[idx]

    @staticmethod
    def backward(ctx, grad_output):
        input, idx = ctx.saved_tensors
        input.zero_()
        return input.index_put_((idx,), grad_output, accumulate=True), None

@profile
def transition_indexing_lowmem(device, word2state):
    C = 2 ** 14
    spw = 512
    V = 10000

    start = torch.randn(C, device=device)
    start.requires_grad = True
    transition = torch.randn(C, C, device=device)
    transition.requires_grad = True
    emission = torch.randn(C, V, device=device)
    emission.requires_grad = True

    batch, time = 16, 32
    timem1 = time - 1

    text = torch.from_numpy(np.random.choice(
        V, size=(16,32))
    ).to(device).long()
    clamped_states = word2state[text]

    log_potentials = Index2.apply(
        transition,
        clamped_states[:,:-1,:,None],
        clamped_states[:,1:,None,:],
    )
    init = start[clamped_states[:,0]]
    init = Index1.apply(start, clamped_states[:,0])
    obs = Index2.apply(emission, clamped_states[:,:,:,None], text[:,:,None,None])
    log_potentials[:,0] += init.unsqueeze(-1)
    log_potentials += obs[:,1:].transpose(-1, -2)
    log_potentials[:,0] += obs[:,0]
    #if wandb.run.mode == "dryrun":
        #print(f"total clamp time: {timep.time() - start_clamp}")
    loss = log_potentials.sum()
    loss.backward()

transition_indexing_lowmem(device, word2state)

# test indexing

C = 2 ** 14
spw = 512
V = 10000

start = torch.randn(C, device=device)
start.requires_grad = True
transition = torch.randn(C, C, device=device)
transition.requires_grad = True
emission = torch.randn(C, V, device=device)
emission.requires_grad = True

batch, time = 16, 32
timem1 = time - 1

text = torch.from_numpy(np.random.choice(
    V, size=(16,32))
).to(device).long()
clamped_states = word2state[text]

def grad_index_old(emission, start, transition, clamped_states):
    log_potentials = transition[
        clamped_states[:,:-1,:,None],
        clamped_states[:,1:,None,:],
    ]
    init = start[clamped_states[:,0]]
    init = start[clamped_states[:,0]]
    obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
    log_potentials[:,0] += init.unsqueeze(-1)
    log_potentials += obs[:,1:].transpose(-1, -2)
    log_potentials[:,0] += obs[:,0]
    #if wandb.run.mode == "dryrun":
        #print(f"total clamp time: {timep.time() - start_clamp}")
    loss = log_potentials.sum()
    loss.backward()

def grad_index_new(emission, start, transition, clamped_states):
    log_potentials = Index2.apply(
        transition,
        clamped_states[:,:-1,:,None],
        clamped_states[:,1:,None,:],
    )
    init = start[clamped_states[:,0]]
    init = Index1.apply(start, clamped_states[:,0])
    obs = Index2.apply(emission, clamped_states[:,:,:,None], text[:,:,None,None])
    log_potentials[:,0] += init.unsqueeze(-1)
    log_potentials += obs[:,1:].transpose(-1, -2)
    log_potentials[:,0] += obs[:,0]
    #if wandb.run.mode == "dryrun":
        #print(f"total clamp time: {timep.time() - start_clamp}")
    loss = log_potentials.sum()
    loss.backward()


grad_index_new(emission, start, transition, clamped_states)
e0 = emission.grad.clone()
s0 = start.grad.clone()
t0 = transition.grad.clone()
emission.grad.zero_()
start.grad.zero_()
transition.grad.zero_()
grad_index_old(emission, start, transition, clamped_states)
e1 = emission.grad.clone()
s1 = start.grad.clone()
t1 = transition.grad.clone()

print(torch.allclose(e0, e1))
print(torch.allclose(s0, s1))
print(torch.allclose(t0, t1))

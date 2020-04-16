

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
C = 2 ** 15
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

@profile
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

emit_old(C, V, spw, word2state, device)


@profile
def transition_indexing(device, word2state):
    C = 2 ** 15
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

    clamped_states = torch.from_numpy(np.random.choice(
        C, size=(16,32,spw))
    ).to(device)

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

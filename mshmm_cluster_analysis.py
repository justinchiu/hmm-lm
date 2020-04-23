
# hmm state analysis
# limited to sentences of length <= 90
 
from collections import Counter

import torch
import torchtext

from torch_scatter import scatter_add

from datasets.ptb import PennTreebank, BucketIterator

import numpy as np

from utils import Pack

count_paths = [
    "mshmm-k16384-spw128-nc128-counts.pth",
    "mshmm-k32768-spw256-nc128-counts.pth",
    "mshmm-k65536-spw512-nc128-counts.pth",
    "mshmm-k32768-spw512-nc64-counts.pth",
    "mshmm-k32768-spw1024-nc32-counts.pth",
]

I = None
for stuff in count_paths:

    path, counts = torch.load(stuff)

    chp = torch.load(path)

    config = chp["args"]

    device = torch.device("cuda:0")
    config.device = device
    config.timing = 0
    config.chp_theta = 0

    TEXT = torchtext.data.Field(batch_first = True)
    train, valid, test = PennTreebank.splits(
        TEXT,
        newline_eos = True,
    )

    TEXT.build_vocab(train)
    V = TEXT.vocab

    from models.mshmmlm import MshmmLm
    model = MshmmLm(V, config)
    model.to(device)
    model.load_state_dict(chp["model"])

    nc = model.config.num_clusters
    cluster_count = torch.zeros(nc, model.states_per_word, device=device, dtype=torch.long)
    w2c = model.word2cluster
    cluster_count.index_add_(0, w2c, counts)

    print(f"k {config.num_classes} spw {config.states_per_word} nb {config.num_clusters}")

    state_usage = (cluster_count > 0).sum(-1)
    if nc != 128:
        C, _ = state_usage.sort()
    elif I is None:
        C, I = state_usage.sort()
    else:
        C = state_usage[I]
    print("State counts per cluster")
    print(C.tolist())
    print("Total number of states used / number of states")
    print(f"{C.sum()} / {config.num_classes}")

    #import pdb; pdb.set_trace()

"""
# old, make sure I matches everything else
def words_in_cluster(w2c, c, V):
    return [V.itos[x] for x in (w2c == c).nonzero().squeeze().tolist()]

num_words = []
for c in range(nc):
    num_words.append(len([model.V.itos[x] for x in (w2c == c).nonzero().squeeze(-1).tolist()]))
num_words = torch.tensor(num_words)
print("Num words in cluster")
print(num_words[I].tolist())

wc = Counter(dict(model.word_counts))
word_counts = torch.LongTensor(
    [wc[x] for x in range(len(V))]
).to(device)

cluster_word_counts = scatter_add(word_counts, w2c)
print("Word counts per cluster")
print(cluster_word_counts[I].tolist())

"""

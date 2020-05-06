
# hmm state analysis
# limited to sentences of length <= 90
 
from collections import Counter

import torch
import torchtext

from torch_scatter import scatter_add

from datasets.lm import PennTreebank
from datasets.data import BucketIterator

import numpy as np

from utils import Pack

_, hmm_counts = torch.load("hmm-k1024-counts.pth")
hmm_state_counts = (hmm_counts > 0).sum(-1)
bin_hmm_counts = hmm_counts > 0
hmm_pairwise = bin_hmm_counts.float() @ bin_hmm_counts.float().t()

"""

count_paths = [
    "mshmm-k16384-spw128-nc128-counts.pth",
    "mshmm-k32768-spw256-nc128-counts.pth",
    "mshmm-k65536-spw512-nc128-counts.pth",
    #"mshmm-k32768-spw512-nc64-counts.pth",
    "mshmm-k32768-spw1024-nc32-counts.pth",
    "mshmm-k16384-spw512-nc32-counts.pth",
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

    word_counts = Counter(dict(model.word_counts))

    print("state usage for words")
    bin_counts = counts > 0
    state_usage_word = (counts > 0).sum(-1)
    words = ["to", "do", "n't", "now", "october", "newspaper", "country", "from"]
    for word in words:
        idx = V[word]
        print(f"{word}: {state_usage_word[idx]}")

    singleton_clusters = (
        (w2c[:,None] == torch.arange(model.config.num_clusters).to(device)).sum(0) == 1
    ).nonzero().squeeze().tolist()
    singleton_words = []
    for cluster in singleton_clusters:
        word = (w2c == cluster).nonzero().squeeze().item()
        singleton_words.append(V.itos[word])
    if singleton_words:
        print("singleton words")
        print(singleton_words)

    # check least used clusters and words
    # avoid shadowing because of complicated logic above
    cluster_counts, idx = state_usage.sort()
    worst_cluster = idx[0]
    print(f"worst cluster {worst_cluster} count {cluster_counts[0]}")
    worst_words = (w2c == worst_cluster).nonzero().squeeze(-1).tolist()
    print(worst_words)
    print(f"worst words (word, count, occ): {[(V.itos[x], state_usage_word[x].item(), word_counts[x]) for x in worst_words]}")

    # comparison to 1k dense HMM
    delta = hmm_state_counts > state_usage_word
    print(f"num words where 1K HMM uses more states: {delta.sum().item()} / {len(V)}")


    bin_counts_f = bin_counts.float()
    for c in range(nc):
        mask = w2c == c
        pairwise = bin_counts_f @ bin_counts_f.t()
        #import pdb; pdb.set_trace()
"""

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


# cluster analysis II
 
stuff = "mshmm-k32768-spw1024-nc32-counts.pth"
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


state_usage = (cluster_count > 0).sum(-1)
C, I = state_usage.sort()
word_counts = Counter(dict(model.word_counts))

bin_counts = counts > 0
state_usage_word = (counts > 0).sum(-1)

singleton_clusters = (
    (w2c[:,None] == torch.arange(model.config.num_clusters).to(device)).sum(0) == 1
).nonzero().squeeze().tolist()
singleton_words = []
for cluster in singleton_clusters:
    word = (w2c == cluster).nonzero().squeeze().item()
    singleton_words.append(V.itos[word])

# comparison to 1k dense HMM
delta = state_usage_word - hmm_state_counts
print(f"num words where 1K HMM uses more states: {(delta > 0).sum().item()} / {len(V)}")
print(f"Delta > 100: {(delta > 100).sum().item()}")
print(f"Delta > 50: {(delta > 50).sum().item()}")
print(f"Delta > 25: {(delta > 25).sum().item()}")

# (state_usage_word - hmm_state_counts).topk(100)
Vp, Ip = hmm_pairwise.sort(-1, descending=True)
x = torch.zeros(len(V), device=device, dtype=torch.bool)
recalls = torch.empty(len(V))
for w in range(len(V)):
    # compare against other words in cluster
    other_words_in_cluster = w2c == w2c[w]
    num_words = other_words_in_cluster.sum()
    x.fill_(0)[Ip[w, :num_words]] = True
    recall = (x * other_words_in_cluster).sum().float() / num_words
    recalls[w] = recall

print(f"Brown cluster recall in HMM state occupancy: min {recalls.min().item():.2f} mean {recalls.mean().item():.2f} median {recalls.median().item():.2f} max {recalls.max().item():.2f}")
print(f"Number of words with > .3 recall: {(recalls > 0.3).sum()}")
print(f"Number of words with 0 recall: {(recalls == 0).sum()}")
#import pdb; pdb.set_trace()


bin_counts_f = bin_counts.float()
for c in range(nc):
    mask = w2c == c
    pairwise = bin_counts_f @ bin_counts_f.t()
    #import pdb; pdb.set_trace()

#import pdb; pdb.set_trace()
meh = (hmm_state_counts - state_usage_word)[hmm_state_counts < state_usage_word].abs().float().mean()


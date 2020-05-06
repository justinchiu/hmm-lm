
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

count_paths = [
    "hmm-k1024-counts.pth",
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

    print(f"k {config.num_classes}")
    state_usage = (counts > 0).sum(-1)

    C, I = state_usage.sort()
    import pdb; pdb.set_trace()

    print("State counts per word")
    print(C.tolist())
    print("Total number of states used / number of states")
    print(f"{C.sum()} / {config.num_classes}")

    word_counts = Counter(dict(model.word_counts))

    print("state usage for words")
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


# hmm state analysis
# limited to sentences of length <= 90
 
from collections import Counter

import torch
import torchtext
from datasets.ptb import PennTreebank, BucketIterator

import numpy as np

from utils import Pack

shmm8_path, shmm8_counts = torch.load("shmm-k1024-spw128-nc8-counts.pth")
# run hmm 16
#shmm8_path, shmm8_counts = torch.load("shmm-k1024-spw64-nc16-counts.pth")
shmm4_path, shmm4_counts = torch.load("shmm-k1024-spw256-nc4-counts.pth")
hmm_path, hmm_counts = torch.load("hmm-k1024-counts.pth")

shmm8_chp = torch.load(shmm8_path)

config = shmm8_chp["args"]

device = torch.device("cuda:0")
config.device = device
config.timing = 0

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = PennTreebank.splits(
    TEXT,
    newline_eos = True,
)

TEXT.build_vocab(train)
V = TEXT.vocab

#from models.shmmlm import ShmmLm
#shmm8 = ShmmLm(V, config)
from models.dshmmlm import DshmmLm
shmm8 = DshmmLm(V, config)
shmm8.to(device)
shmm8.load_state_dict(shmm8_chp["model"])

word_counts = Counter(dict(shmm8.word_counts))

num_nonzero_hmm = (hmm_counts > 0).sum(-1)
num_nonzero_shmm8 = (shmm8_counts > 0).sum(-1)
num_nonzero_shmm4 = (shmm4_counts > 0).sum(-1)

# basic statistics
print("State count statistics")
print(f"hmm mean (min, med, max): {num_nonzero_hmm.float().mean().item()} ({num_nonzero_hmm.min().item()}, {num_nonzero_hmm.median().item()}, {num_nonzero_hmm.max().item()})")
print(f"shmm4 mean (min, med, max): {num_nonzero_shmm4.float().mean().item()} ({num_nonzero_shmm4.min().item()}, {num_nonzero_shmm4.median().item()}, {num_nonzero_shmm4.max().item()})")
print(f"shmm8 mean (min, med, max): {num_nonzero_shmm8.float().mean().item()} ({num_nonzero_shmm8.min().item()}, {num_nonzero_shmm8.median().item()}, {num_nonzero_shmm8.max().item()})")

# largest numbers of unique states
print("Topk number of clusters for words")
print(f"Topk nonzero hmm: {num_nonzero_hmm.topk(20)}")
print(f"Topk nonzero shmm-4: {num_nonzero_shmm4.topk(20)}")
print(f"Topk nonzero shmm-8: {num_nonzero_shmm8.topk(20)}")

print("Many words only have a single state")
print(f"words w single state hmm: {(num_nonzero_hmm == 1).sum().item()}")
print(f"words w single state shmm-4: {(num_nonzero_shmm4 == 1).sum().item()}")
print(f"words w single state shmm-8: {(num_nonzero_shmm8 == 1).sum().item()}")

# construct word to count dicts
word2counts = {}
for word, count in shmm8.word_counts:
    word2counts[V.itos[word]] = Pack(
        count = count,
        hmm_count = num_nonzero_hmm[word].item(),
        shmm4_count = num_nonzero_shmm4[word].item(),
        shmm8_count = num_nonzero_shmm8[word].item(),
    )

def print_wc(word):
    counts = word2counts[word]
    wc = counts.count
    hc = counts.hmm_count
    s4c = counts.shmm4_count
    s8c = counts.shmm8_count
    print(f"{word} | # {wc} | h {hc} | s4 {s4c} | s8 {s8c}")

print("word | # occurrences | hmm unique states | shmm-4 unique states | shmm-8 unique states")
words = ["a", "to", "the", "very", "brown", "aer", "run", "quickly"]
for word in words:
    print_wc(word)

# plot token rank (by word count) vs state counts (instead of word counts)
flatlist = [
    (
        word,
        counts.count,
        counts.hmm_count,
        counts.shmm4_count,
        counts.shmm8_count,
    ) for word, counts in word2counts.items()
]
sortedlist = sorted(flatlist, key=lambda x: x[1], reverse=True)
words = [x[0] for x in sortedlist]
rank_hmm = [x[2] for x in sortedlist]
rank_shmm4 = [x[3] for x in sortedlist]
rank_shmm8 = [x[4] for x in sortedlist]

import chart_studio.plotly as py
import plotly.graph_objects as go
import scipy.signal

for name, y in [
    ("hmm (174)", rank_hmm),
    ("shmm4 (174)", rank_shmm4),
    ("shmm8 (186)", rank_shmm8),
    #("shmm16 (188)", rank_shmm8),
]:
    #x = np.arange(len(y)),
    x = words[:100]
    y = y[:100]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x = x,
        y = y,
        name=name,
    ))
    fig.update_layout(
        title = f"{name} Rank (word count) vs # unique states",
        xaxis_title = "Words by rank (word count)",
        yaxis_title = "# unique states with nonzero count (# states as max of state marg)",
        xaxis_tickangle = -45,
    )
    fig.update_yaxes(range=[0, 300])
    py.plot(
        fig,
        filename=f"{name} Rank v uniq states", 
        sharing="public", auto_open=False,
    )
#import pdb; pdb.set_trace()

# compare to brown clusters?
 

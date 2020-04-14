
# hmm state analysis
# limited to sentences of length <= 90
 
from collections import Counter

import torch
import torchtext
from datasets.ptb import PennTreebank, BucketIterator

import numpy as np

from utils import Pack

shmm8_path, shmm8_counts = torch.load("shmm-k1024-spw128-nc8-counts.pth")
shmm4_path, shmm4_counts = torch.load("shmm-k1024-spw256-nc4-counts.pth")

#shmm16_path, shmm16_counts = torch.load("shmm-k1024-spw64-nc16-counts.pth")
shmm8_path, shmm8_counts = torch.load("shmm-k1024-spw64-nc16-counts.pth")

shmm8_chp = torch.load(shmm8_path)
shmm4_chp = torch.load(shmm4_path)

use8 = False
print(
    "brown HMM 8 clusters"
    if use8
    else "brown HMM 4 clusters"
)

config = shmm8_chp["args"] if use8 else shmm4_chp["args"]

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

if use8:
    from models.dshmmlm import DshmmLm
    shmm8 = DshmmLm(V, config)
    shmm8.to(device)
    shmm8.load_state_dict(shmm8_chp["model"])
    model = shmm8

    nc = model.config.num_clusters
    cluster_count = torch.zeros(nc, model.states_per_word, device=device, dtype=torch.long)
    w2c = model.word2cluster
    cluster_count.index_add_(0, w2c, shmm8_counts)
else:
    from models.shmmlm import ShmmLm
    """
    # try buggy 8 one
    shmm4_chp = torch.load(
        "wandb_checkpoints/shmm_k1024_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb8_nc0_ncs0_spc0/176648_5.30.pth"
    )
    """
    config = shmm4_chp["args"]
    config.timing = 0
    shmm4 = ShmmLm(V, config)
    shmm4.to(device)
    shmm4.load_state_dict(shmm4_chp["model"])
    model = shmm4

    nc = model.config.num_clusters
    cluster_count = torch.zeros(nc, model.states_per_word, device=device, dtype=torch.long)
    w2c = model.word2cluster
    w2c = torch.tensor(
        [
            model.word2cluster[c] if c in model.word2cluster else nc-1
            for c in range(len(V))
        ],
        dtype=torch.long, device=device,
    )
    cluster_count.index_add_(0, w2c, shmm4_counts)

print((cluster_count > 0).sum(-1).tolist())

def words_in_cluster(w2c, c, V):
    return [V.itos[x] for x in (w2c == c).nonzero().squeeze().tolist()]

num_words = []
for c in range(nc):
    num_words.append(len([model.V.itos[x] for x in (w2c == c).nonzero().squeeze().tolist()]))
print(num_words)

import pdb; pdb.set_trace()

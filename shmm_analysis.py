
import sys

import math
import time

from collections import Counter

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

import torch_struct as ts
from models.autoregressive import Autoregressive

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchtext
from datasets.ptb import PennTreebank, BucketIterator

from args import get_args

from utils import set_seed, get_config, get_mask_lengths
from models.lstmlm import LstmLm
from models.fflm import FfLm

import chart_studio.plotly as py
import plotly.graph_objects as go

import pandas as pd


chp_path = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b1024_adamw_lr0.01_c5_tw_nas0_pw1_asbrown/8337_4.90.pth"
chp_path2 = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b1024_adamw_lr0.01_c5_tw_nas0_pw1_asuniform/8335_4.97.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
config.device = device

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = PennTreebank.splits(
    TEXT,
    newline_eos = True,
)

TEXT.build_vocab(train)
V = TEXT.vocab

train_iter, valid_iter, text_iter = BucketIterator.splits(
    (train, valid, test),
    #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
    batch_size = 4,
    device = device,
    sort_key = lambda x: len(x.text),
)

from models.shmmlm import ShmmLm

model = ShmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

chp2 = th.load(chp_path2)
# chp["args"] will have the args eventually...
#config = get_args()
config2 = chp["args"]
model2 = ShmmLm(V, config2)
model2.to(device)
model2.load_state_dict(chp2["model"])

model.eval()
model2.eval()

un_counts = model.counts[:,4:]
counts = un_counts / un_counts.sum(0)

word_counts = model.word_counts
wc = Counter(dict(word_counts))

cols = (counts > 1e-2).float().sum(0)
avg_counts = cols.mean(0)
max_counts = cols.max(0)
min_counts = cols.min(0)

# just counts
cols_np = cols.cpu().numpy()
py.plot(
    [go.Violin(y=cols_np)],
    filename="counts", sharing="public", auto_open=False,
)

# word counts vs number > 1e-2
w_counts = np.empty((len(V)-4,))
i = 0
for w in range(4, len(V)):
    w_counts[i] = wc[w]
    i += 1
py.plot(
    [go.Scatter(y=cols_np, x=w_counts, mode="markers",)],
    filename="wcounts_v_counts", sharing="public", auto_open=False,
)

wc_pd = pd.DataFrame({
    "w": w_counts,
    "c": cols_np,
}, columns = ["w", "c"],)
wc_pd_w = wc_pd.groupby("w")["c"]

py.plot(
    [
        go.Scatter(
            y=wc_pd_w.mean().values,
            x=wc_pd_w.mean().index.values,
            mode="lines",
        ),
        go.Scatter(
            y=wc_pd_w.max().values,
            x=wc_pd_w.max().index.values,
            mode="lines",
            line=dict(color="firebrick", dash="dot"),
        ),
        go.Scatter(
            y=wc_pd_w.min().values,
            x=wc_pd_w.min().index.values,
            mode="lines",
            line=dict(color="royalblue", dash="dot"),
        ),
    ],
    filename="wcounts_v_counts_err", sharing="public", auto_open=False,
)


# max emit prob vs # occurrence

logp_x_z = model.mask_emission(model.emission_logits, model.word2state).detach()
# get max_z log p(x | z) for each x
logp, idxs = logp_x_z.max(0)

py.plot(
    [
        go.Scatter(
            y=logp.exp()[4:].cpu().numpy(),
            x=w_counts,
            mode="markers",
        ),
    ],
    filename="wcounts_v_maxprob", sharing="public", auto_open=False,
)

# dominated states vs # occurrence
dom_counts = th.zeros(len(model.V)).index_add(0, logp_x_z.max(-1).indices.cpu(), th.ones(model.C)).numpy()
py.plot(
    [
        go.Scatter(
            y=dom_counts[4:],
            x=w_counts,
            mode="markers",
        ),
    ],
    filename="wcounts_v_domcounts", sharing="public", auto_open=False,
)
print(f"There are {(dom_counts == 0).sum()} not in the top of any state")

dom2_counts = th.zeros(len(model.V)).index_add(0,
    logp_x_z.topk(2, -1).indices.cpu().view(-1), th.ones(model.C * 2)).numpy()
py.plot(
    [
        go.Scatter(
            y=dom2_counts[4:],
            x=w_counts,
            mode="markers",
        ),
    ],
    filename="wcounts_v_dom2counts", sharing="public", auto_open=False,
)
print(f"There are {(dom2_counts == 0).sum()} not in the top2 of any state")

dom5_counts = th.zeros(len(model.V)).index_add(0,
    logp_x_z.topk(5, -1).indices.cpu().view(-1), th.ones(model.C * 5)).numpy()
py.plot(
    [
        go.Scatter(
            y=dom2_counts[4:],
            x=w_counts,
            mode="markers",
        ),
    ],
    filename="wcounts_v_dom2counts", sharing="public", auto_open=False,
)
print(f"There are {(dom5_counts == 0).sum()} not in the top5 of any state")

# what are some words that are not emit by any state?

# entropy
with th.no_grad():
    trans1 = th.distributions.Categorical(logits=model.transition_logits.log_softmax(-1)).entropy()
    trans2 = th.distributions.Categorical(logits=model2.transition_logits.log_softmax(-1)).entropy()

    # emission entropy
    le1 = model.mask_emission(model.emission_logits, model.word2state)
    e1 = le1 * le1.exp()
    e1[e1 != e1] = 0
    emit1 = -e1.sum(-1)
    le2 = logits=model2.mask_emission(model2.emission_logits, model2.word2state)
    e2 = le2 * le2.exp()
    e2[e2 != e2] = 0
    emit2 = -e2.sum(-1)

    print(f"Trans entropy mean {trans1.mean()} | max {trans1.max()} | min {trans1.min()}")
    print(f"Trans2 entropy mean {trans2.mean()} | max {trans2.max()} | min {trans2.min()}")
    print(f"Emit entropy mean {emit1.mean()} | max {emit1.max()} | min {emit1.min()}")
    print(f"Emit2 entropy mean {emit2.mean()} | max {emit2.max()} | min {emit2.min()}")

import pdb; pdb.set_trace()


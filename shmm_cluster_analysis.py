
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


# 128
chp_path = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/38402_4.90.pth"
# 64
chp_path2 = "wandb_checkpoints/shmm_k16384_wps512_spw256_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/42254_4.90.pth"
# uniform 128
chp_path3 = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asuniform_nc0_ncs0_spc0/30724_4.98.pth"

chp = th.load(chp_path)
chp2 = th.load(chp_path2)
chp3 = th.load(chp_path3)
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

# one sentence at a time
train_iter, valid_iter, text_iter = BucketIterator.splits(
    (train, valid, test),
    #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
    batch_size = 1,
    batch_size_fn = lambda new, count, sofar: count,
    #batch_size_fn = f,
    device = device,
    sort_key = lambda x: len(x.text),
)

from models.shmmlm import ShmmLm

model = ShmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

model2 = ShmmLm(V, chp2["args"])
model2.to(device)
model2.load_state_dict(chp2["model"])

model3 = ShmmLm(V, chp3["args"])
model3.to(device)
model3.load_state_dict(chp3["model"])

word_counts = dict(model.word_counts)

model.eval()
model2.eval()
model3.eval()


# model 1 has 121 clusters,
state_counts = model.state_counts[:121*128]
state_counts2 = model2.state_counts
state_counts3 = model3.state_counts

# histograms of state counts
doplot = False
if doplot:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=state_counts.cpu().numpy(),
        ),
    )
    # this doesn't work
    #fig.update_layout(
        #xaxis_type="log",
        #yaxis_type="log",
    #)
    py.plot(
        fig,
        filename="state counts brown 128",
        sharing="public", auto_open=False,
    )

    # histograms of state counts for brown 64
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=state_counts2.cpu().numpy(),
        ),
    )
    # this doesn't work
    #fig.update_layout(
        #xaxis_type="log",
        #yaxis_type="log",
    #)
    py.plot(
        fig,
        filename="state counts brown 64",
        sharing="public", auto_open=False,
    )

    # histograms of state counts for uniform 128
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=state_counts3.cpu().numpy(),
        ),
    )
    # this doesn't work
    #fig.update_layout(
        #xaxis_type="log",
        #yaxis_type="log",
    #)
    py.plot(
        fig,
        filename="state counts uniform 128",
        sharing="public", auto_open=False,
    )

    # histograms of state counts comparing brown + uniform 128
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=state_counts3.cpu().numpy(),
        ),
    )
    fig.add_trace(
        go.Histogram(
            x=state_counts.cpu().numpy(),
        ),
    )
    # this doesn't work
    #fig.update_layout(
        #xaxis_type="log",
        #yaxis_type="log",
    #)
    py.plot(
        fig,
        filename="state counts brown + uniform 128",
        sharing="public", auto_open=False,
    )

    # histograms of state counts comparing brown + uniform 128 + brown 64
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=state_counts.cpu().numpy(),
        ),
    )
    fig.add_trace(
        go.Histogram(
            x=state_counts2.cpu().numpy(),
        ),
    )
    # this doesn't work
    #fig.update_layout(
        #xaxis_type="log",
        #yaxis_type="log",
    #)
    py.plot(
        fig,
        filename="state counts brown 128 + brown 64",
        sharing="public", auto_open=False,
    )

def tail(sc, n):
    return (sc > n).sum().float() / sc.nelement()

ns = [5, 10, 50, 100, 200, 500]
scs = [state_counts, state_counts2, state_counts3]
namedict = {
    0: "brown-128",
    1: "brown-64",
    2: "uniform",
}

for n in ns:
    for i, sc in enumerate(scs):
        with th.no_grad():
            print(f"{namedict[i]}: {tail(sc, n):.3f} states have more than {n} counts")

# free up memory for subsequent analysis
del state_counts
del state_counts2
del state_counts3
del model.state_counts
del model2.state_counts
del model3.state_counts

# sparsity analysis

def e(m):
    return m.mask_emission(m.emission_logits, m.word2state)
def t(m):
    return m.mask_transition(m.transition_logits)
def nz(x):
    return x[x > 0]

def H(lp):
    h = lp.exp() * lp
    h[h != h] = 0
    return -h.sum(-1)

models = [model, model2, model3]
for i, m in enumerate(models):
    with th.no_grad():
        le = e(m)
        hs = H(le)
        print(f"{namedict[i]} emission entropy mean (min,med,max): {hs.mean().item():.2f} ({hs.min().item():.2f}, {hs.median().item():.2f}, {hs.max().item():.2f})")
        del le
        del hs
        lt = t(m)
        hs = H(lt)
        print(f"{namedict[i]} transition entropy mean (min,med,max): {hs.mean().item():.2f} ({hs.min().item():.2f}, {hs.median().item():.2f}, {hs.max().item():.2f})")
        del lt
        del hs

for i, m in enumerate([model, model2]):
    with th.no_grad():
        # cluster level
        # emission is annoying to implement, need scatter_lse
        spw = m.states_per_word
        C = m.C
        lt = t(m).view(C // spw, spw, C // spw, spw).logsumexp(1).logsumexp(-1).log_softmax(-1)
        hs = H(lt)
        print(f"{namedict[i]} cluster transition entropy mean (min,med,max): {hs.mean().item():.2f} ({hs.min().item():.2f}, {hs.median().item():.2f}, {hs.max().item():.2f})")

        if doplot:
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=lt.exp().cpu().numpy(),
                ),
            )
            py.plot(
                fig,
                filename=f"{namedict[i]} cluster transition (source, target)",
                sharing="public", auto_open=False,
            )

import pdb; pdb.set_trace()

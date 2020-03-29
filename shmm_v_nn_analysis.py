
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


chp_path = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/38402_4.90.pth"
chp_path2 = "wandb_checkpoints/ff_k16384_wps512_spw128_ed256_d256_dp0.3_tdp0_cdp0_tokens_b1024_adamw_lr0.001_c5.0_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/92647_5.08.pth"
chp_path3 = "wandb_checkpoints/lstm_k16384_wps512_spw128_ed256_d256_dp0.3_tdp0_cdp0_tokens_b512_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/93550_4.60.pth"

#chp_path = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/38402_4.90.pth"
#chp_path2 = "wandb_checkpoints/shmm_k16384_wps512_spw256_ed256_d256_dp0_tdp0.5_cdp1_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/42254_4.90.pth"

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

from models.fflm import FfLm

chp2 = th.load(chp_path2)
# chp["args"] will have the args eventually...
#config = get_args()
config2 = chp2["args"]
model2 = FfLm(V, config2)
model2.to(device)
model2.load_state_dict(chp2["model"])

from models.lstmlm import LstmLm

chp3 = th.load(chp_path3)
# chp["args"] will have the args eventually...
#config = get_args()
config3 = chp3["args"]
model3 = LstmLm(V, config3)
model3.to(device)
model3.load_state_dict(chp3["model"])

model.eval()
model2.eval()
model3.eval()

f_out = open("compare_wordprobs.log", "w")
f_out.write("word (hmm, ff, lstm)\n")

data1 = []
data2 = []
data3 = []
probs_and_counts_ff = th.zeros((2, len(V)), device=device)
nextprobs_and_counts_ff = th.zeros((2, len(V)), device=device)
probs_and_counts_lstm = th.zeros((2, len(V)), device=device)
nextprobs_and_counts_lstm = th.zeros((2, len(V)), device=device)
probs_and_pos_ff = th.zeros((2, 50), device=device)
probs_and_pos_lstm = th.zeros((2, 50), device=device)
with th.no_grad():
    for batch in valid_iter:
        mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
        # hmm
        l1 = model.score(batch.text, mask=mask, lengths=lengths)
        log_pots = model.log_potentials(batch.text)
        m, a, b, log_m = model.fb(log_pots, mask)
        log_unary_marginals = th.cat([        
            log_m[:,0,None].logsumexp(-2),
            log_m.logsumexp(-1),          
        ], 1)

        x0 = batch.text[0,0]
        z0 = model.word2state[batch.text[0,0]]
        lpx_z0 = model.mask_emission(model.emission_logits, model.word2state)[z0, x0]
        lpx_0 = (model.start[z0] + lpx_z0).logsumexp(0)

        lpx = a.logsumexp(-1).squeeze()
        lpx[0] = lpx_0
        lpx[1:] = (lpx[1:] - lpx[:-1])

        lpx2 = model2.lpx(batch.text, mask, lengths).squeeze()
        lpx3 = model3.lpx(batch.text, mask, lengths).squeeze()

        # ff
        l2 = model2.score(batch.text, mask=mask, lengths=lengths)
        # lstm
        l3 = model3.score(batch.text, mask=mask, lengths=lengths)

        al1 = -(l1.evidence / n_tokens).item()
        al2 = -(l2.evidence / n_tokens).item()
        al3 = -(l3.evidence / n_tokens).item()

        text = batch.text.squeeze()
        # probability of word vs word count
        probs_and_counts_ff[0].index_add_(
            0,
            text,
            th.ones(lpx.shape[0], device=device),
        )
        probs_and_counts_ff[1].index_add_(
            0,
            text,
            lpx - lpx2,
        )
        probs_and_counts_lstm[0].index_add_(
            0,
            text,
            th.ones(lpx.shape[0], device=device),
        )
        probs_and_counts_lstm[1].index_add_(
            0,
            text,
            lpx - lpx3,
        )
        # probability of word vs preceeding word count
        nextprobs_and_counts_ff[0].index_add_(
            0,
            text[:-1],
            th.ones(lpx.shape[0], device=device)[:-1],
        )
        nextprobs_and_counts_ff[1].index_add_(
            0,
            text[:-1],
            (lpx - lpx2)[1:],
        )
        nextprobs_and_counts_lstm[0].index_add_(
            0,
            text[:-1],
            th.ones(lpx.shape[0], device=device)[:-1],
        )
        nextprobs_and_counts_lstm[1].index_add_(
            0,
            text[:-1],
            (lpx - lpx3)[1:],
        )
        f_out.write(" ".join([
            f"{model.V.itos[x]} ({p1:.2f}, {p2:.2f}, {p3:.2f})"
            for x, p1, p2, p3
            in zip(batch.text[0].tolist(), lpx.tolist(), lpx2.tolist(), lpx3.tolist())
        ]))
        f_out.write("\n")

        """
        print(" ".join([model.V.itos[x] for x in batch.text[0].tolist()]))

        print(f"hmm\t{al1:.2f}")
        print(f"ff\t{al2:.2f}")
        print(f"lstm\t{al3:.2f}")

        import pdb; pdb.set_trace()
        """

        data1.append((al1, n_tokens.item()))
        data2.append((al2, n_tokens.item()))
        data3.append((al3, n_tokens.item()))

        T = min(probs_and_pos_ff.shape[1], lengths.item()-1)
        # probability of word vs position in sentence
        probs_and_pos_ff[0,:T] += 1
        probs_and_pos_ff[1,:T] += (lpx - lpx2)[:T]
        probs_and_pos_lstm[0,:T] += 1
        probs_and_pos_lstm[1,:T] += (lpx - lpx3)[:T]

f_out.close()

doplot = False

data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)

if doplot:
    py.plot(
        [
            go.Scatter(
                y=data1[:,0] - data3[:,0],
                x=data1[:,1],
                mode="markers",
            ),
        ],
        filename="hmm - lstm v length", sharing="public", auto_open=False,
    )
    py.plot(
        [
            go.Scatter(
                y=data1[:,0] - data2[:,0],
                x=data1[:,1],
                mode="markers",
            ),
        ],
        filename="hmm - ff v length", sharing="public", auto_open=False,
    )

probs_and_counts_ff = probs_and_counts_ff.cpu().numpy()
nextprobs_and_counts_ff = nextprobs_and_counts_ff.cpu().numpy()
probs_and_counts_lstm = probs_and_counts_lstm.cpu().numpy()
nextprobs_and_counts_lstm = nextprobs_and_counts_lstm.cpu().numpy()

np.save("probs_and_counts_ff", probs_and_counts_ff)
np.save("nextprobs_and_counts_ff", nextprobs_and_counts_ff)
np.save("probs_and_counts_lstm", probs_and_counts_lstm)
np.save("nextprobs_and_counts_lstm", nextprobs_and_counts_lstm)

probs_and_pos_ff = probs_and_pos_ff.cpu().numpy()
probs_and_pos_lstm = probs_and_pos_lstm.cpu().numpy()
np.save("probs_and_pos_ff", probs_and_pos_ff)
np.save("probs_and_pos_lstm", probs_and_pos_lstm)

doplot = False
if doplot:
    py.plot(
        [
            go.Scatter(
                y=probs_and_counts_ff[1] / probs_and_counts_ff[0],
                x=probs_and_counts_ff[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log hmm prob div ff prob)", sharing="public", auto_open=False,
    )
    py.plot(
        [
            go.Scatter(
                y=nextprobs_and_counts_ff[1] / nextprobs_and_counts_ff[0],
                x=nextprobs_and_counts_ff[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log next hmm prob div ff prob) div counts", sharing="public", auto_open=False,
    )

    py.plot(
        [
            go.Scatter(
                y=probs_and_counts_lstm[1] / probs_and_counts_lstm[0],
                x=probs_and_counts_lstm[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log hmm prob div lstm prob)", sharing="public", auto_open=False,
    )
    py.plot(
        [
            go.Scatter(
                y=nextprobs_and_counts_lstm[1] / nextprobs_and_counts_lstm[0],
                x=nextprobs_and_counts_lstm[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log next hmm prob div lstm prob) div counts", sharing="public", auto_open=False,
    )

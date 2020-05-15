
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
from datasets.lm import PennTreebank
from datasets.data import BucketIterator, BPTTIterator

from args import get_args

from utils import set_seed, get_config, get_mask_lengths
from models.lstmlm import LstmLm
from models.fflm import FfLm

import chart_studio.plotly as py
import plotly.graph_objects as go

# BPTT
# 16k shmm unstructured 0.75 nb128
chp_path = "wandb_checkpoints/ptb_bptt_shmm_k16384_wps512_spw128_tspw64_ed256_d256_dp0_tdp0.75_cdp1_sdp0_dtunstructured_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc1/88983_5.01.pth"
# 16k shmm unstructured 0.5 nb128
chp_path = "wandb_checkpoints/ptb_bptt_shmm_k16384_wps512_spw128_tspw64_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtunstructured_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc1/19974_4.99.pth"
# 16k shmm unstructured 0.25 nb128
chp_path = "wandb_checkpoints/ptb_bptt_shmm_k16384_wps512_spw128_tspw64_ed256_d256_dp0_tdp0.25_cdp1_sdp0_dtunstructured_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc1/14527_5.02.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
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

# one sentence at a time
if config.iterator == "bptt":
    train_iter, valid_iter, text_iter = BPTTIterator.splits(
        (train, valid, test),
        #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
        #batch_size = 512,
        batch_size = config.eval_bsz,
        #batch_size_fn = lambda new, count, sofar: count,
        #batch_size_fn = f,
        device = device,
        bptt_len = config.bptt,
    )
else:
    train_iter, valid_iter, text_iter = BucketIterator.splits(
        (train, valid, test),
        #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
        #batch_size = 512,
        batch_size = config.eval_bsz,
        #batch_size_fn = lambda new, count, sofar: count,
        #batch_size_fn = f,
        device = device,
        sort_key = lambda x: len(x.text),
    )

from models.shmmlm import ShmmLm
config.dropout_type = "none"
model = ShmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

use_train = False
biter = train_iter if use_train else valid_iter

counts = th.zeros(len(model.V) * model.states_per_word, device=device, dtype=th.long)
post_counts = th.zeros(len(model.V) * model.states_per_word, device=device, dtype=th.long)
with th.no_grad():
    model.train(False)
    start = model.start()
    emission = model.mask_emission(model.emission_logits(), model.word2state)

    if model.C > 2 ** 15:
        start = start.cpu()
        emission = emission.cpu()
        # blocked transition
        num_blocks = 128
        block_size = model.C // num_blocks
        next_state_proj = model.next_state_proj.weight
        transition = th.empty(model.C, model.C, device=th.device("cpu"), dtype=emission.dtype)
        for s in range(0, model.C, block_size):
            states = range(s, s+block_size)
            x = model.trans_mlp(model.dropout(model.state_emb.weight[states]))
            y = (x @ next_state_proj.t()).log_softmax(-1)
            transition[states] = y.to(transition.device)
    else:
        transition = model.mask_transition(model.transition_logits())

    lpz, last_states = None, None

    Hs = []

    for batch in biter:
        mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
        text = batch.text

        if config.iterator == "bptt" and lpz is not None and last_states is not None:
            tmp = lpz[:,:,None] + transition[last_states]
            start = tmp.logsumexp(1)

        # hmm
        log_pots = model.clamp(
            batch.text, start, transition, emission, model.word2state,
        ).to(model.device)

        log_m, alphas = model.fb(log_pots, mask)

        # log p(zT)
        N, T = text.shape
        idx = th.arange(N, device=model.device)
        alpha_T = alphas[lengths-1, idx]

        last_words = text[idx, lengths-1]
        last_states = model.word2state[last_words]
        lpz = alpha_T.log_softmax(-1)

        x0 = batch.text[:,0]
        z0 = model.word2state[batch.text[:,0]]
        lpx_z0 = emission[z0, x0[:,None]]
        if start.ndim == 1:
            lpx_0 = (start[z0] + lpx_z0).logsumexp(-1)
        else:
            lpx_0 = (start.gather(-1, z0) + lpx_z0).logsumexp(-1)

        # log p(x)
        # alphas: T x N x |Z|
        lpx = alphas.logsumexp(-1).squeeze()
        lpx[0] = lpx_0
        lpx[1:] = (lpx[1:] - lpx[:-1])

        log_unary_marginals = th.cat([
            log_m[:,0,None].logsumexp(-2),
            log_m.logsumexp(-1),
        ], 1)
        mask = text != model.V["<pad>"]
        text = text[mask]
        I = log_unary_marginals.max(-1).indices[mask]
        counts.index_add_(0, text * model.states_per_word + I, th.ones_like(I, dtype=th.long))

        H_post = th.distributions.Categorical(logits=log_unary_marginals).entropy()
        Hs.append(H_post)

counts = counts.view(len(model.V), model.states_per_word)
savepath = f"{config.dataset}-{config.iterator}-shmm-k{model.C}-spw{model.states_per_word}-nc{model.config.num_clusters}-dp{config.transition_dropout}-{config.dropout_type}-counts-train{use_train}.pth"
th.save((chp_path, counts), savepath)
print(f"Saved counts to {savepath}")

Hs = th.tensor([x for H in Hs for x in H.view(-1).tolist()])
print("Posterior entropy")
print(f"{Hs.min().item():.3f}, {Hs.median().item():.3f}, {Hs.mean().item():.3f}, {Hs.max().item():.3f}")



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


chp_path = "wandb_checkpoints/shmm_k1024_wps512_spw256_ed256_d256_dp0_tdp0.25_cdp1_sdp0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb4_nc0_ncs0_spc0/39239_5.16.pth"
# this one is broken
#chp_path = "wandb_checkpoints/shmm_k1024_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb8_nc0_ncs0_spc0/176648_5.30.pth"
chp_path = "wandb_checkpoints/dshmm_k1024_wps512_spw128_ed256_d256_dp0_tdp0.0_cdp1_sdp0_dtstate_tokens_b128_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nb8_nc0_ncs0_spc0/113458_5.23.pth"
chp_path = "wandb_checkpoints/dshmm_k1024_wps512_spw64_ed256_d256_dp0_tdp0.0_cdp1_sdp0_dtstate_tokens_b128_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nb16_nc0_ncs0_spc0/64843_5.24.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
config.device = device
config.timing = 0

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
    batch_size = 96,
    #batch_size_fn = lambda new, count, sofar: count,
    #batch_size_fn = f,
    device = device,
    sort_key = lambda x: len(x.text),
)

from models.dshmmlm import DshmmLm
#from models.shmmlm import ShmmLm
config.dropout_type = "none"
#model = ShmmLm(V, config)
model = DshmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

counts = th.zeros(len(model.V) * model.states_per_word, device=device, dtype=th.long)
with th.no_grad():
    for batch in train_iter:
        mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
        text = batch.text
        # hmm
        #l1 = model.score(batch.text, mask=mask, lengths=lengths)
        log_pots = model.log_potentials(text)
        m, a, b, log_m = model.fb(log_pots, mask)
        del log_pots, m, a, b
        log_unary_marginals = th.cat([        
            log_m[:,0,None].logsumexp(-2),
            log_m.logsumexp(-1),          
        ], 1)
        mask = text != model.V["<pad>"]
        text = text[mask]
        I = log_unary_marginals.max(-1).indices[mask]
        counts.index_add_(0, text * model.states_per_word + I, th.ones_like(I, dtype=th.long))

counts = counts.view(len(model.V), model.states_per_word)
th.save((chp_path, counts), f"shmm-k1024-spw{model.states_per_word}-nc{model.config.num_clusters}-counts.pth")


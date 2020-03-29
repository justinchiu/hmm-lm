
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
        import pdb; pdb.set_trace()


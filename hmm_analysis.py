
import sys

import math
import time

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

import argstrings

from utils import set_seed, get_config, get_args, get_mask_lengths
from models.lstmlm import LstmLm
from models.fflm import FfLm

args = get_args(argstrings.lm_args)
print(args)

device = th.device("cuda:0")

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

from models.hmmlm import HmmLm

for model_config, chp_path in zip(
    [
        get_config("configs/hmm-d256-k256-oldres.yaml", device),
        get_config("configs/hmm-d256-k512-oldres.yaml", device),
        get_config("configs/hmm-d256-k1024-oldres.yaml", device),
    ],
    [
        "checkpoints/hmm_b4_d256_k256_oldres/13_5.34.pth",
        "checkpoints/hmm_b4_d256_k512_oldres/16_5.20.pth",
        "checkpoints/hmm_b4_d256_k1024_oldres/11_5.15.pth",
    ],
):
    model = HmmLm(V, model_config)
    #from models.arhmmlm import ArHmmLm
    #model = ArHmmLm(V, model_config)
    model.to(device)

    #chp = th.load("checkpoints/hmm_b4_d256_k256_oldres/13_5.34.pth")
    #chp = th.load("checkpoints/hmm_b4_d256_k512_oldres/16_5.20.pth")
    chp = th.load(chp_path)
    model.load_state_dict(chp["model"])

    den = model_config.num_classes * 5

    print(model.transition.nelement())

    print((model.transition.exp() < 1 / den).sum())
    print(float((model.transition.exp() < 1 / den).sum()) / float(model.transition.nelement()))

    #print((model.transition.exp() > 0.1 ).sum())
    #print(float((model.transition.exp() > 0.1).sum()) / float(model.transition.nelement()))

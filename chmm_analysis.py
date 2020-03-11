
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

from args import get_args

from utils import set_seed, get_config, get_mask_lengths
from models.lstmlm import LstmLm
from models.fflm import FfLm

chp_path = "wandb_checkpoints/chmm_k8192_wps512_spw128_ed512_d512_dp0.0_tdp0.5_tokens_b1024_adamw_lr0.01_c5.0_tw/12967_5.01.pth"
chp_path2 = "wandb_checkpoints/chmm_k8192_wps512_spw128_ed512_d512_dp0.0_tdp0.5_tokens_b1024_adamw_lr0.01_c5.0_tw/12950_5.02.pth"

chp = th.load(chp_path)
chp2 = th.load(chp_path2)
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

from models.chmmlm import ChmmLm

model = ChmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

model2 = ChmmLm(V, config)
model2.to(device)
model2.load_state_dict(chp2["model"])
import pdb; pdb.set_trace()

den = model_config.num_classes * 5

print(model.transition.nelement())

print((model.transition.exp() < 1 / den).sum())
print(float((model.transition.exp() < 1 / den).sum()) / float(model.transition.nelement()))

#print((model.transition.exp() > 0.1 ).sum())
#print(float((model.transition.exp() > 0.1).sum()) / float(model.transition.nelement()))

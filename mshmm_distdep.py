
import readline
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

chp_path = "wandb_checkpoints/shmm_k1024_wps512_spw256_ed256_d256_dp0_tdp0.25_cdp1_sdp0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb4_nc0_ncs0_spc0/39239_5.16.pth"
# this one is broken
#chp_path = "wandb_checkpoints/shmm_k1024_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_tokens_b256_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb8_nc0_ncs0_spc0/176648_5.30.pth"
chp_path = "wandb_checkpoints/dshmm_k1024_wps512_spw128_ed256_d256_dp0_tdp0.0_cdp1_sdp0_dtstate_tokens_b128_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nb8_nc0_ncs0_spc0/113458_5.23.pth"
chp_path = "wandb_checkpoints/dshmm_k1024_wps512_spw64_ed256_d256_dp0_tdp0.0_cdp1_sdp0_dtstate_tokens_b128_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nb16_nc0_ncs0_spc0/64843_5.24.pth"
chp_path = "wandb_checkpoints/mshmm_k65536_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0/45333_4.81.pth"
# 32k 128spw 256b
chp_path = "wandb_checkpoints/mshmm_k32768_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb256_nc0_ncs0_spc0/173948_4.86.pth"
# 16k 128spw 128b
chp_path = "wandb_checkpoints/mshmm_k16384_wps512_spw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0/146552_4.89.pth"
# 32k 1024spw 32b
chp_path = "wandb_checkpoints/mshmm_k32768_wps512_spw1024_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb32_nc0_ncs0_spc0/85495_4.85.pth"
# 32k 256spw 128b
chp_path = "wandb_checkpoints/mshmm_k32768_wps512_spw256_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0/68111_4.83.pth"
# 65k 512spw 128b
chp_path = "wandb_checkpoints/mshmm_k65536_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0/45333_4.81.pth"
# 32k 512 spw 64b
chp_path = "wandb_checkpoints/mshmm_k32768_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb64_nc0_ncs0_spc0/94177_4.86.pth"
# 16k 512 spw 32b
chp_path = "wandb_checkpoints/mshmm_k16384_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb32_nc0_ncs0_spc0/170998_4.92.pth"

# BPTT
# oops! not shmm
# 16k mshmm state 0.25 nb128
chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw96_ed256_d256_dp0_tdp0.25_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/33897_4.89.pth"
# 16k mshmm state 0.5 nb128
chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw64_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/89588_4.90.pth"
# 16k mshmm state 0.75 nb128
#chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw32_ed256_d256_dp0_tdp0.75_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/90799_5.03.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
#device = th.device("cpu")
config.device = device
config.timing = 0
config.chp_theta = 0
if "state" not in config:
    config.state = "ind"
    config.emit = "word"

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
        batch_size = 1,
        #batch_size_fn = lambda new, count, sofar: count,
        #batch_size_fn = f,
        device = device,
        bptt_len = 32,
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

from models.mshmmlm import MshmmLm
#from models.shmmlm import ShmmLm
config.dropout_type = "none"
#model = ShmmLm(V, config)
model = MshmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

train_iter.sort = False
train_iter.shuffle = False

use_train = True
biter = train_iter if use_train else valid_iter

model.train(False)

lpz, last_states = None, None
for batch in biter:
    mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
    text = batch.text

    print(" ".join(V.itos[x] for x in text[0].tolist()))

    start = model.start()
    emission = model.mask_emission(model.emission_logits(), model.word2state)
    transition = model.mask_transition(model.transition_logits())
    if config.iterator == "bptt" and lpz is not None and last_states is not None:
        tmp = lpz[:,:,None] + transition[last_states]
        start = tmp.logsumexp(1)

    import pdb; pdb.set_trace()
    # hmm
    log_pots = model.clamp(
        batch.text, start, transition, emission, model.word2state,
    ).to(model.device)

    log_m, alphas = model.fb_test(log_pots, mask)

    x0 = batch.text[:,0]                                     
    z0 = model.word2state[batch.text[:,0]]                   
    lpx_z0 = emission[z0, x0[:,None]]                        
    if start.ndim == 1:        
        lpx_0 = (start[z0] + lpx_z0).logsumexp(-1)
    else:
        lpx_0 = (start.gather(-1, z0) + lpx_z0).logsumexp(-1)

    # log p(x)
    lpx = alphas.logsumexp(-1).squeeze()
    lpx[0] = lpx_0
    lpx[1:] = (lpx[1:] - lpx[:-1])

    # log p(zT)
    N, T = text.shape
    idx = th.arange(text.shape[0], device=model.device)
    alpha_T = alphas[lengths-1, idx]

    last_words = text[idx, lengths-1]              
    last_states = model.word2state[last_words]
    lpz = alpha_T.log_softmax(-1)

    import pdb; pdb.set_trace()

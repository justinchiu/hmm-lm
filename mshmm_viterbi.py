
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

from utils import set_seed, get_config, get_mask_lengths, get_name
from models.lstmlm import LstmLm
from models.fflm import FfLm

import chart_studio.plotly as py
import plotly.graph_objects as go

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb_max",
    "hmm_runners/viterbi.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

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
#chp_path = "wandb_checkpoints/mshmm_k65536_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0/45333_4.81.pth"
# 32k 512 spw 64b
#chp_path = "wandb_checkpoints/mshmm_k32768_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb64_nc0_ncs0_spc0/94177_4.86.pth"
# 16k 512 spw 32b
#chp_path = "wandb_checkpoints/mshmm_k16384_wps512_spw512_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtNone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb32_nc0_ncs0_spc0/170998_4.92.pth"

# BPTT
# oops! not shmm
# 16k mshmm state 0.25 nb128
#chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw96_ed256_d256_dp0_tdp0.25_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/33897_4.89.pth"
# 16k mshmm state 0.5 nb128
#chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw64_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/89588_4.90.pth"
# 16k mshmm state 0.75 nb128
#chp_path = "wandb_checkpoints/ptb_bptt_mshmm_k16384_wps512_spw128_tspw32_ed256_d256_dp0_tdp0.75_cdp1_sdp0_dtstate_wd0_tokens_b16_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns1_fc1/90799_5.03.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
config.device = device
config.timing = 0
config.chp_theta = 0
if "dataset" not in config:
    # means older than when wikitext was added
    config.dataset = "ptb"
    config.reset_eos = 0
    config.no_shuffle_train = 0
    config.flat_clusters = 0
    config.iterator = "bucket"
    # old default
    config.train_spw = config.states_per_word // 2

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = PennTreebank.splits(
    TEXT,
    newline_eos = True,
)

TEXT.build_vocab(train)
V = TEXT.vocab

# use pos dataset
dataset = PennTreebank(".data/PTB/ptb.nopunct.txt", TEXT)

# one sentence at a time?
if config.iterator == "bptt":
    raise NotImplementedError
    biter = BPTTIterator(
        dataset,
        batch_size = config.eval_bsz,
        device = device,
        bptt_len = config.bptt,
    )
else:
    biter = BucketIterator(
        dataset,
        batch_size = 1,
        batch_size_fn = lambda x,y,z: 1,
        device = device,
        sort_key = lambda x: len(x.text),
        shuffle = False,
        sort = False,
    )
    # DONT SORT

config.dropout_type = "none"
if config.model == "mshmm":
    from models.mshmmlm import MshmmLm
    model = MshmmLm(V, config)
elif config.model == "shmm":
    from models.shmmlm import ShmmLm
    model = ShmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

fb_max = foo.get_fb_max(model.states_per_word)

use_train = False


counts = th.zeros(len(model.V) * model.states_per_word, device=device, dtype=th.long)
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
    viterbi_sequences = []

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
        """
        with th.enable_grad():
            print(lengths)
            log_m = ts.LinearChain(ts.MaxSemiring).marginals(log_pots, lengths)
        print(ts.LinearChain(ts.MaxSemiring).sum(log_pots, lengths))
        print(log_m.nonzero())
        N, T, C, _ = log_m.shape
        """

        # make sure bsz 1
        # transpose to time x batch x left x right
        max_margs = fb_max(log_pots.transpose(0, 1).transpose(-1, -2).clone())
        T, N, C, _ = max_margs.shape

        for n in range(N):
            """
            best = log_m[n, :lengths[n]]
            parts = best.nonzero()
            viterbi_sequence0 = [parts[0,2].item()] + parts[:,1].tolist()
            """
            viterbi_sequence = [
                max_margs[0,0].max(-1).values.argmax(-1).item()
            ] + max_margs[:,0].max(-2).values.max(-1).indices.tolist()
            viterbi_sequences.append(viterbi_sequence)

# save viterbi sequences
import pickle
name = get_name(config)
with open(f"viterbi_output/{name}.viterbi.pkl", "wb") as f:
    pickle.dump(viterbi_sequences, f)

txtf = Path(f"viterbi_output/{name}.viterbi.txt")
txtf.write_text("\n".join([
    " ".join(map(str, sequence))
    for sequence in viterbi_sequences
]))

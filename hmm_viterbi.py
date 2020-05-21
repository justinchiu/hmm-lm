
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
from datasets.lm import Wsj
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

chp_path = "wandb_checkpoints/wsj_bucket_hmm_k45_wps512_spw128_tspw64_ed512_d512_dp0_tdp0.0_cdp1_sdp0_dtnone_wd0_tokens_b512_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc0_eword_schmm/85692_5.61.pth"

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
if "char_dim" not in config:
    config.char_dim = 0
    config.emit = "word"
    config.emit_dims = None
    config.num_highway = 0
    config.state = "ind"

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = Wsj.splits(
    TEXT,
    newline_eos = True,
)

TEXT.build_vocab(train)
V = TEXT.vocab

# use pos dataset
dataset = train

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
elif config.model == "hmm":
    from models.hmmlm import HmmLm
    model = HmmLm(V, config)
model.to(device)
model.load_state_dict(chp["model"])

fb_max = foo.get_fb_max(model.C)

with th.no_grad():
    model.train(False)
    start = model.start
    emission = model.emission
    transition = model.mask_transition(model.transition_logits())

    lpz = None
    viterbi_sequences = []

    for batch in biter:
        mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
        text = batch.text

        if config.iterator == "bptt" and lpz is not None:
            tmp = lpz[:,:,None] + transition
            start = tmp.logsumexp(1)                                                 

        # hmm
        log_pots = model.clamp(
            batch.text, start, transition, emission,
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
        assert N == 1

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

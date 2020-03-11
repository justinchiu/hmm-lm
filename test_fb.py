
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

from utils import set_seed, get_config, get_args, get_mask_lengths, log_eye
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

model_config = get_config("configs/hmm-d256-k128-oldres.yaml", device)

from models.hmmlm import HmmLm

for _ in range(10):
    model = HmmLm(V, model_config)
    #from models.arhmmlm import ArHmmLm
    #model = ArHmmLm(V, model_config)
    model.to(device)

    batch = next(iter(train_iter))
    mask, lengths, n_tokens = get_mask_lengths(batch.text, V)

    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )

    log_pots1 = log_potentials.detach()
    log_pots1.requires_grad = True
    log_pots2 = log_potentials.detach()
    log_pots2.requires_grad = True

    # testing surrogate loss gradient
    log_px = ts.LinearChain().sum(log_pots1)
    log_px.sum().backward()
    (model.fb(log_pots2)[0].detach() * log_pots2).sum().backward()
    print((log_pots1.grad - log_pots2.grad).abs().max())
    print(th.allclose(log_pots1.grad, log_pots2.grad, atol=1e-6))

    marginals = ts.LinearChain().marginals(log_pots1, lengths=lengths)
    m, a, b = model.fb(log_pots2, mask=mask)
    print((marginals - m).abs().max())
    assert th.allclose(marginals, m, atol=1e-6)

    del log_pots1
    del log_pots2
    del marginals
    del m
    del a
    del b
    del log_px

    # full integration test
    model.zero_grad()

    # method 1: compute evidence explicitly
    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )
    ts.LinearChain().sum(log_potentials, lengths=lengths).sum().backward()
    grad1 = model.state_emb.grad.clone()

    # method 2: surrogate loss
    model.zero_grad()
    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )
    (model.fb(log_potentials, mask)[0].detach() * log_potentials).sum().backward()
    grad2 = model.state_emb.grad.clone()

    # method 3: feed edge marginals as grad_out
    model.zero_grad()
    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )
    th.autograd.backward(log_potentials, model.fb(log_potentials, mask)[0])
    grad3 = model.state_emb.grad.clone()

    # method 4: surrogate loss no length
    model.zero_grad()
    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )
    (model.fb(log_potentials)[0].detach() * log_potentials).sum().backward()
    grad4 = model.state_emb.grad.clone()

    # method 1b: no lengths
    model.zero_grad()
    log_potentials = ts.LinearChain.hmm(
        transition = model.transition,
        emission = model.emission,
        init = model.start,
        observations = batch.text,
        semiring = model.semiring,
    )
    ts.LinearChain().sum(log_potentials).sum().backward()
    grad1b = model.state_emb.grad.clone()

    print((grad1b - grad4).abs().max())
    assert th.allclose(grad1b, grad4, atol=1e-6)
    assert th.allclose(grad1, grad2, atol=1e-6)
    assert th.allclose(grad1, grad3, atol=1e-6)
    del grad1
    del grad1b
    del grad2
    del grad3
    del grad4
    del model
    del log_potentials

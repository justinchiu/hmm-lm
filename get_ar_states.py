
import math

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
from models.hmmlm import HmmLm


def report(loss, n, prefix):
    print(f"{prefix}: log_prob = {loss:.2f} | xent(word) = {-loss / n:.2f} | ppl = {math.exp(-loss / n):.2f}")

def count_params(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

def get_best_model(path):
    # fix list comp later
    return min(
        [
            (float(x.name.split("_")[-1][:-4]), x)
            for x in path.iterdir()
            if x.suffix == ".pth" and x.name != "states.pth"
        ],
        key = lambda x: x[0],
    )[1]

def _loop(args, V, iter, model, valid_iter=None):
    env = th.no_grad
    states = []
    words = []
    idxs = []
    with env():
        for i, batch in enumerate(iter):
            model.train(False)
            mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
            emb_x = model.dropout(model.emb(batch.text))
            rnn_o, _ = model.lstm(emb_x)
            for idx, l in enumerate(lengths.tolist()):
                states.append(rnn_o[idx, :l].cpu())
                words.append(batch.text[idx, :l].cpu())
                idxs.append(batch.idxs[idx])
    #return th.cat(states, 0), th.cat(words, 0)
    return states, words, idxs


def main():
    args = get_args(argstrings.lm_args)
    print(args)

    set_seed(args.seed)

    device = th.device("cpu" if args.devid < 0 else f"cuda:{args.devid}")

    TEXT = torchtext.data.Field(batch_first = True)
    train, valid, test = PennTreebank.splits(TEXT, newline_eos=False)
    TEXT.build_vocab(train)
    V = TEXT.vocab

    train_iter, valid_iter, text_iter = BucketIterator.splits(
        (train, valid, test),
        #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
        batch_size = args.bsz,
        device = device,
        sort = False,
    )

    model_config = get_config(args.model_config, device)
    model = (LstmLm(V, model_config)
        if model_config.type == "lstm" else HmmLm(V, model_config))
    model.to(device)
    print(model)

    save_path = Path(f"checkpoints/{model_config.name}")
    path = get_best_model(save_path)
    model.load_state_dict(th.load(path)["model"])

    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable: {num_params}, {num_trainable_params}")

    states, words, idxs = _loop(
        args, V, train_iter, model,
    )
    valid_states, valid_words, valid_idxs = _loop(
        args, V, valid_iter, model,
    )

    th.save({
        "states": states,
        "words": words,
        "idxs": idxs,
        "valid_states": valid_states,
        "valid_words": valid_words,
        "valid_idxs": valid_idxs,
    }, f"checkpoints/{model_config.name}/states.pth")


if __name__ == "__main__":
    main()

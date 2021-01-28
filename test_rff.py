
import time as timep

import sys

import math
import time

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

import torch_struct as ts
from models.autoregressive import Autoregressive

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import torchtext
from datasets.lm import PennTreebank, WikiText2, Wsj
from datasets.data import BucketIterator, BPTTIterator
#from torchtext.data import BPTTIterator

from args import get_args

from utils import set_seed, get_config, get_name, get_mask_lengths
from utils import Pack
from utils import plot_counts

from models.lstmlm import LstmLm
from models.fflm import FfLm

#th.autograd.set_detect_anomaly(True)

valid_schedules = ["reducelronplateau"]

WANDB_STEP = -1

BEST_VALID = -math.inf
PREV_SAVE = None

def max_diff(log_p):
    P = log_p.exp()
    n = P.shape[0]
    xy = [(x,y) for x in range(n) for y in range(x, n)]
    diff = th.tensor([(P[x] - P[y]).abs().max() for x,y in xy])
    print(f"Max diff < 0.01: {(diff < 0.01).sum()} / {diff.shape[0]}")
    #import pdb; pdb.set_trace()


def update_best_valid(
    valid_losses, valid_n, model, optimizer, scheduler, name,
):
    global WANDB_STEP
    global BEST_VALID
    global PREV_SAVE
    if valid_losses.evidence > BEST_VALID:
        BEST_VALID = valid_losses.evidence


def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    # cap loss otherwise overflow
    #loss = loss if loss > -1e7 else -1e7
    str_list = [
        f"{prefix}: log_prob = {loss:.2f}",
        f"xent(word) = {-loss / n:.2f}",
        f"ppl = {math.exp(-loss / n):.2f}",
    ]
    if elbo is not None:
        str_list.append(f"elbo = {elbo / n:.2f}")
    total_time = None
    if start_time is not None:
        total_time = time.time() - start_time
        str_list.append(f"total_time = {total_time:.2f}s")
    print(" | ".join(str_list))
    return total_time

def count_params(model):
    return (
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

def eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    lpz, last_states = None, None
    with th.no_grad():
        for i, batch in enumerate(iter):
            model.train(False)
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0
            mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
            if args.iterator != "bptt":
                lpz, last_states = None, None
            losses, lpz, _ = model.score(
                batch.text,
                lpz=lpz, last_states = last_states,
                mask=mask, lengths=lengths,
            )
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n


def train_loop(
    args, V, iter, model,
    parameters, optimizer, scheduler,
    valid_iter=None,
    verbose=False,
):
    global WANDB_STEP

    noise_scales = np.linspace(1, 0, args.noise_anneal_steps)
    total_ll = 0
    total_elbo = 0
    n = 0
    # check is performed at end of epoch outside loop as well
    checkpoint = len(iter) // (args.num_checks - 1)
    with th.enable_grad():
        lpz = None
        last_states = None
        for i, batch in enumerate(iter):
            model.train(True)
            WANDB_STEP += 1
            optimizer.zero_grad()

            text = batch.textp1 if "lstm" in args.model else batch.text
            if args.iterator == "bucket":
                lpz = None
                last_states = None

            mask, lengths, n_tokens = get_mask_lengths(text, V)
            if model.timing:
                start_forward = timep.time()

            # check if iterator == bptt

            losses, _, _= model.score_rff(
                text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)

            if model.timing:
                print(f"forward time: {timep.time() - start_forward}")
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens

            loss = -losses.loss / n_tokens
            if model.timing:
                start_backward = timep.time()
            loss.backward()

            # copy gradient
            grad1 = model.next_state_emb.grad.detach().clone()
            #import pdb; pdb.set_trace()

            optimizer.zero_grad()
            print(model.next_state_emb.grad.max())
            #import pdb; pdb.set_trace()
            losses_old, lpz, last_states = model.score(
                text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)

            loss_old = -losses_old.loss / n_tokens
            if model.timing:
                start_backward = timep.time()
            loss_old.backward()

            grad2 = model.next_state_emb.grad.detach().clone()

            import pdb; pdb.set_trace()

            #print(model.state_emb.grad.max(), model.state_emb.grad.min())
            #print(model.next_state_emb.grad.max(), model.next_state_emb.grad.min())
            #import pdb; pdb.set_trace()

            if model.timing:
                print(f"backward time: {timep.time() - start_backward}")
            clip_grad_norm_(parameters, args.clip)
            if args.schedule not in valid_schedules:
                # sched before opt since we want step = 1?
                # this is how huggingface does it
                scheduler.step()
            optimizer.step()

            if verbose and i % args.report_every == args.report_every - 1:
                report(
                    Pack(evidence = total_ll, elbo = total_elbo),
                    n,
                    f"Train batch {i}",
                )

            if valid_iter is not None and i % checkpoint == checkpoint-1:
                v_start_time = time.time()
                #eval_fn = cached_eval_loop if args.model == "mshmm" else eval_loop
                #valid_losses, valid_n  = eval_loop(
                #valid_losses, valid_n  = cached_eval_loop(
                if args.model == "mshmm" or args.model == "factoredhmm":
                    if args.num_classes > 2 ** 15:
                        eval_fn = mixed_cached_eval_loop
                    else:
                        eval_fn = cached_eval_loop
                elif args.model == "hmm" or args.model == "lhmm":
                    eval_fn = cached_eval_loop
                else:
                    eval_fn = eval_loop
                valid_losses, valid_n  = eval_fn(
                    args, V, valid_iter, model,
                )
                report(valid_losses, valid_n, "Valid eval", v_start_time)

                update_best_valid(
                    valid_losses, valid_n, model, optimizer, scheduler, args.name)

                scheduler.step(valid_losses.evidence)

                # remove this later?
                if args.log_counts > 0 and args.keep_counts > 0:
                    # TODO: FACTOR OUT
                    counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
                    c, v = counts.shape
                    #cg4 = counts > 1e-4
                    #cg3 = counts > 1e-3
                    cg2 = counts > 1e-2

                    del cg2
                    del counts

    return Pack(evidence = total_ll, elbo = total_elbo), n


def main():
    global WANDB_STEP
    #args = get_args(argstrings.lm_args)
    args = get_args()
    print(args)

    set_seed(args.seed)

    device = th.device("cpu" if args.devid < 0 else f"cuda:{args.devid}")
    args.device = device
    aux_device = th.device("cpu" if args.aux_devid < 0 else f"cuda:{args.aux_devid}")
    args.aux_device = aux_device

    if args.dbg_double:
        # test higher precision
        th.set_default_tensor_type(th.cuda.DoubleTensor)
        # reminder: never use 'as th' again

    TEXT = torchtext.data.Field(batch_first = True)
    ## DBG
    #TEXT = torchtext.data.Field(batch_first = True, lower=True)

    if args.dataset == "ptb":
        Dataset = PennTreebank
    elif args.dataset == "wikitext103":
        Dataset = WikiText103
    elif args.dataset == "wikitext2":
        # shuffling the articles is annoying
        Dataset = WikiText2
        #Dataset = torchtext.datasets.WikiText2
    elif args.dataset == "wsj":
        Dataset = Wsj

    train, valid, test = Dataset.splits(
        TEXT,
        newline_eos = True,
    )

    TEXT.build_vocab(train)
    V = TEXT.vocab

    def batch_size_tokens(new, count, sofar):
        return max(len(new.text), sofar)
    def batch_size_sents(new, count, sofar):
        return count

    if args.iterator == "bucket":
        # independent sentences...bad
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            #batch_size = args.bsz,
            device = device,
            sort_key = lambda x: len(x.text),
            batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,
        )
    elif args.iterator == "bptt":
        #train_iter, valid_iter, text_iter = BPTTIterator.splits(
        train_iter, valid_iter, test_iter = BPTTIterator.splits(
            (train, valid, test),
            batch_sizes = [args.bsz, args.eval_bsz, args.eval_bsz],
            #batch_size = args.bsz,
            device = device,
            bptt_len = args.bptt,
            sort = False,
        )
    else:
        raise ValueError(f"Invalid iterator {args.iterator}")

    if args.no_shuffle_train:
        train_iter.shuffle = False

    """
    args = get_config(args.model_config, device)
    config = Pack(args.items() | args.items())
    # for now, until migrate to argparse
    config.model = args.type
    """
    name = get_name(args)
    import tempfile
    args.name = name

    model = None
    if args.model == "lstm":
        model = LstmLm(V, args)
    elif args.model == "hmm":
        from models.hmmlm import HmmLm
        model = HmmLm(V, args)
    elif args.model == "lhmm":
        from models.lhmmlm import LHmmLm
        model = LHmmLm(V, args)
    elif args.model == "ff":
        model = FfLm(V, args)
    elif args.model == "arhmm":
        from models.arhmmlm import ArHmmLm
        model = ArHmmLm(V, args)
    elif args.model == "poehmm":
        from models.poehmmlm import PoeHmmLm
        model = PoeHmmLm(V, args)
    elif args.model == "chmm":
        from models.chmmlm import ChmmLm
        model = ChmmLm(V, args)
    elif args.model == "dhmm":
        from models.dhmmlm import DhmmLm
        model = DhmmLm(V, args)
    elif args.model == "shmm":
        from models.shmmlm import ShmmLm
        model = ShmmLm(V, args)
    elif args.model == "dhmm":
        from models.dhmmlm import DhmmLm
        model = DhmmLm(V, args)
    elif args.model == "mshmm":
        from models.mshmmlm import MshmmLm
        model = MshmmLm(V, args)
    elif args.model == "dshmm":
        from models.dshmmlm import DshmmLm
        model = DshmmLm(V, args)
    elif args.model == "factoredhmm" and args.param == "scalar":
        # hack for scalar
        from models.newhmmlm import FactoredHmmLm
        model = FactoredHmmLm(V, args)
    elif args.model == "factoredhmm":
        from models.factoredhmmlm import FactoredHmmLm
        model = FactoredHmmLm(V, args)
    else:
        raise ValueError("Invalid model type")
    model.to(device)
    print(model)
    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable: {num_params:,}, {num_trainable_params:,}")

    # augment training data
    if "train_features" in args:
        train_features = np.load(
            args.train_features,
            allow_pickle = True,
        )
        raise NotImplementedError

    #DEBUG
    #valid_losses, valid_n = mixed_cached_eval_loop(args, V, valid_iter, model)
    #import pdb; pdb.set_trace()
    if args.eval_only:
        # uncomment this later
        model.load_state_dict(th.load(args.eval_only)["model"])
        v_start_time = time.time()
        #valid_losses, valid_n = eval_loop(
        #valid_losses, valid_n = cached_eval_loop(
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        #eval_fn = cached_eval_loop if args.model == "mshmm" else eval_loop
        valid_losses, valid_n = eval_fn(
            args, V, valid_iter, model,
        )
        report(valid_losses, valid_n, f"Valid perf", v_start_time)

        t_start_time = time.time()
        test_losses, test_n = eval_fn(
            args, V, test_iter, model,
        )
        report(test_losses, test_n, f"Test perf", t_start_time)

        sys.exit()

    parameters = list(model.parameters())
    if args.optimizer == "adamw":
        optimizer = AdamW(
            parameters,
            lr = args.lr,
            betas = (args.beta1, args.beta2),
            weight_decay = args.wd,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            parameters,
            lr = args.lr,
        )
    if args.schedule == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor = 1. / args.decay,
            patience = args.patience,
            verbose = True,
            mode = "max",
        )
    elif args.schedule == "noam":
        warmup_steps = args.warmup_steps
        def get_lr(step):
            scale = warmup_steps ** 0.5 * min(step ** (-0.5), step * warmup_steps ** (-1.5))
            return args.lr * scale
        scheduler = LambdaLR(
            optimizer,
            get_lr,
            last_epoch=-1,
            verbse = True,
        )
    else:
        raise ValueError("Invalid schedule options")

    # training loop, factor out later if necessary
    for e in range(args.num_epochs):
        start_time = time.time()
        if args.log_counts > 0 and args.keep_counts > 0:
            # reset at START of epoch
            model.state_counts.fill_(0)
        train_losses, train_n = train_loop(
            args, V, train_iter, model,
            parameters, optimizer, scheduler,
            valid_iter = valid_iter if not args.overfit else None,
            verbose = True,
        )
        total_time = report(train_losses, train_n, f"Train epoch {e}", start_time)

        v_start_time = time.time()
        #eval_fn = cached_eval_loop if args.model == "mshmm" else eval_loop
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm" or args.model == "lhmm":
            eval_fn = cached_eval_loop
        else:
            eval_fn = eval_loop
        valid_losses, valid_n  = eval_fn(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid epoch {e}", v_start_time)

        if args.schedule in valid_schedules:
            scheduler.step(
                valid_losses.evidence if not args.overfit else train_losses.evidence)

        update_best_valid(
            valid_losses, valid_n, model, optimizer, scheduler, args.name)


        if args.log_counts > 0 and args.keep_counts > 0:
            # TODO: FACTOR OUT
            # only look at word tokens
            counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
            c, v = counts.shape
            #cg4 = counts > 1e-4
            #cg3 = counts > 1e-3
            cg2 = counts > 1e-2

            # state counts
            # log these once per epoch, then set back to zero
            sc0 = (model.state_counts == 0).sum()
            sc1 = (model.state_counts == 1).sum()
            sc2 = (model.state_counts == 2).sum()
            sc3 = (model.state_counts == 3).sum()
            sc4 = (model.state_counts == 4).sum()
            sc5 = (model.state_counts >= 5).sum()

            del cg2
            del counts

    # won't use best model. Rerun with eval_only
    t_start_time = time.time()
    test_losses, test_n = eval_fn(
        args, V, test_iter, model,
    )
    report(test_losses, test_n, f"Test perf", t_start_time)

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()

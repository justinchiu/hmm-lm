
import sys

import math
import time

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

import torch_struct as ts
from models.autoregressive import Autoregressive

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import torchtext
from datasets.ptb import PennTreebank, BucketIterator

from args import get_args

from utils import set_seed, get_config, get_name, get_mask_lengths
from utils import Pack
from utils import plot_counts

from models.lstmlm import LstmLm
from models.fflm import FfLm

import wandb


valid_schedules = ["reducelronplateau"]

WANDB_STEP = -1

BEST_VALID = -math.inf
PREV_SAVE = None

def update_best_valid(
    valid_losses, valid_n, model, optimizer, scheduler, name,
):
    global WANDB_STEP
    global BEST_VALID
    global PREV_SAVE
    if valid_losses.evidence > BEST_VALID:
        # do not save on dryruns
        if wandb.run.mode == "run":
            save_f = f"wandb_checkpoints/{name}/{WANDB_STEP}_{-valid_losses.evidence / valid_n:.2f}.pth"
            print(f"Saving model to {save_f}")
            Path(save_f).parent.mkdir(parents=True, exist_ok=True)
            th.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": model.config,
            }, save_f)
            if PREV_SAVE is not None:
                Path(PREV_SAVE).unlink()
            PREV_SAVE = save_f

        BEST_VALID = valid_losses.evidence
        wandb.run.summary["best_valid_ppl"] = math.exp(-BEST_VALID / valid_n)
        wandb.run.summary["best_valid_loss"] = BEST_VALID / valid_n


def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    # cap loss otherwise overflow
    loss = loss if loss > -1e7 else -1e7
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

def _loop(
    args, V, iter, model,
    parameters=None, optimizer=None, scheduler=None,
    valid_iter=None,
    verbose=False,
):
    global WANDB_STEP
    noise_scales = np.linspace(1, 0, args.noise_anneal_steps)
    env = th.no_grad if optimizer is None else th.enable_grad
    total_ll = 0
    total_elbo = 0
    n = 0
    # check is performed at end of epoch outside loop as well
    checkpoint = len(iter) // (args.num_checks - 1)
    with env():
        for i, batch in enumerate(iter):
            model.train(optimizer is not None)
            if optimizer is not None:
                WANDB_STEP += 1
                optimizer.zero_grad()

            # set noise scale
            if optimizer is not None and hasattr(model, "noise_scale"):
                noise_scale = noise_scales[
                    min(WANDB_STEP, args.noise_anneal_steps-1)
                ] if args.noise_anneal_steps > 0 else model.init_noise_scale
                model.noise_scale = noise_scale
                wandb.log({
                    "noise_scale": noise_scale,
                }, step=WANDB_STEP)
            elif optimizer is None and hasattr(model, "noise_scale"):
                model.noise_scale = 0

            mask, lengths, n_tokens = get_mask_lengths(batch.text, V)
            losses = model.score(batch.text, mask=mask, lengths=lengths)
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
            if optimizer is not None:
                loss = -losses.loss / n_tokens
                loss.backward()
                clip_grad_norm_(parameters, args.clip)
                if args.schedule not in valid_schedules:
                    # sched before opt since we want step = 1?
                    # this is how huggingface does it
                    scheduler.step()
                optimizer.step()
                wandb.log({
                    "running_training_loss": total_ll / n,
                    "running_training_ppl": math.exp(-total_ll / n),
                }, step=WANDB_STEP)

            if verbose and i % args.report_every == args.report_every - 1:
                report(
                    Pack(evidence = total_ll, elbo = total_elbo),
                    n,
                    f"Train batch {i}",
                )

            if valid_iter is not None and i % checkpoint == checkpoint-1:
                v_start_time = time.time()
                valid_losses, valid_n  = _loop(args, V, valid_iter, model)
                report(valid_losses, valid_n, "Valid eval", v_start_time)
                wandb.log({
                    "valid_loss": valid_losses.evidence / valid_n,
                    "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
                }, step=WANDB_STEP)

                update_best_valid(
                    valid_losses, valid_n, model, optimizer, scheduler, args.name)

                if optimizer is not None:
                    wandb.log({
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=WANDB_STEP)
                scheduler.step(valid_losses.evidence)

                # remove this later?
                if args.log_counts > 0 and args.keep_counts > 0:
                    # TODO: FACTOR OUT
                    counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
                    c, v = counts.shape
                    #cg4 = counts > 1e-4
                    #cg3 = counts > 1e-3
                    cg2 = counts > 1e-2

                    # state counts
                    sc0 = (model.state_counts == 0).sum()
                    sc1 = (model.state_counts == 1).sum()
                    sc2 = (model.state_counts == 2).sum()
                    sc3 = (model.state_counts == 3).sum()
                    sc4 = (model.state_counts == 4).sum()
                    sc5 = (model.state_counts >= 5).sum()

                    wandb.log({
                        #"avgcounts@1e-4": cg4.sum().item() / float(v),
                        #"avgcounts@1e-3": cg3.sum().item() / float(v),
                        "avgcounts@1e-2": cg2.sum().item() / float(v),
                        #"maxcounts@1e-4": cg4.sum(0).max().item() / float(v),
                        #"maxcounts@1e-3": cg3.sum(0).max().item() / float(v),
                        "maxcounts@1e-2": cg2.sum(0).max().item(),
                        #"mincounts@1e-4": cg4.sum(0).min().item() / float(v),
                        #"mincounts@1e-3": cg3.sum(0).min().item() / float(v),
                        "mincounts@1e-2": cg2.sum(0).min().item(),
                        "maxcounts": counts.sum(0).max().item(),
                        "mincounts": counts.sum(0).min().item(),
                        "statecounts=0": sc0,
                        "statecounts=1": sc1,
                        "statecounts=2": sc2,
                        "statecounts=3": sc3,
                        "statecounts=4": sc4,
                        "statecounts>=5": sc5,
                    }, step=WANDB_STEP)
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

    TEXT = torchtext.data.Field(batch_first = True)
    train, valid, test = PennTreebank.splits(
        TEXT,
        newline_eos = True,
    )

    TEXT.build_vocab(train)
    V = TEXT.vocab

    def batch_size_tokens(new, count, sofar):
        return len(new.text) + sofar
    def batch_size_sents(new, count, sofar):
        return count

    train_iter, valid_iter, text_iter = BucketIterator.splits(
        (train, valid, test),
        #batch_size = [args.bsz, args.eval_bsz, args.eval_bsz],
        batch_size = args.bsz,
        device = device,
        sort_key = lambda x: len(x.text),
        batch_size_fn = batch_size_tokens if args.bsz_fn == "tokens" else batch_size_sents,
    )

    """
    args = get_config(args.model_config, device)
    config = Pack(args.items() | args.items())
    # for now, until migrate to argparse
    config.model = args.type
    """
    name = get_name(args)
    wandb.init(project="hmm-lm", name=name, config=args)
    args.name = name

    model = None
    if args.model == "lstm":
        model = LstmLm(V, args)
    elif args.model == "hmm":
        from models.hmmlm import HmmLm
        model = HmmLm(V, args)
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
    else:
        raise ValueError("Invalid model type")
    model.to(device)
    print(model)
    num_params, num_trainable_params = count_params(model)
    print(f"Num params, trainable: {num_params:,}, {num_trainable_params:,}")
    wandb.run.summary["num_params"] = num_params

    # augment training data
    if "train_features" in args:
        train_features = np.load(
            args.train_features,
            allow_pickle = True,
        )
        raise NotImplementedError

    if args.eval_only:
        model.load_state_dict(th.load(args.eval_only)["model"])
        v_start_time = time.time()
        valid_losses, valid_n = _loop(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid perf", v_start_time)
        sys.exit()

    parameters = list(model.parameters())
    optimizer = AdamW(
        parameters,
        lr = args.lr,
        betas = (args.beta1, args.beta2),
        weight_decay = args.wd,
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
        train_losses, train_n = _loop(
            args, V, train_iter, model,
            parameters, optimizer, scheduler,
            valid_iter = valid_iter if not args.overfit else None,
            verbose = True,
        )
        total_time = report(train_losses, train_n, f"Train epoch {e}", start_time)

        v_start_time = time.time()
        valid_losses, valid_n = _loop(args, V, valid_iter, model)
        report(valid_losses, valid_n, f"Valid epoch {e}", v_start_time)

        if args.schedule in valid_schedules:
            scheduler.step(
                valid_losses.evidence if not args.overfit else train_losses.evidence)

        update_best_valid(
            valid_losses, valid_n, model, optimizer, scheduler, args.name)

        wandb.log({
            "train_loss": train_losses.evidence / train_n,
            "train_ppl": math.exp(-train_losses.evidence / train_n),
            "epoch_time": total_time,
            "valid_loss": valid_losses.evidence / valid_n,
            "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
            "best_valid_loss": BEST_VALID / valid_n,
            "best_valid_ppl": math.exp(-BEST_VALID / valid_n),
            "epoch": e,
        }, step=WANDB_STEP)

        if args.log_counts > 0 and args.keep_counts > 0:
            # TODO: FACTOR OUT
            """
            wandb.log({
                "counts": plot_counts(
                    (model.counts / model.counts.sum(0, keepdim=True)).cpu().numpy()
                ),
            }, step=WANDB_STEP)
            """
            # only look at word tokens
            counts = (model.counts / model.counts.sum(0, keepdim=True))[:,4:]
            c, v = counts.shape
            #cg4 = counts > 1e-4
            #cg3 = counts > 1e-3
            cg2 = counts > 1e-2
            wandb.log({
                #"avgcounts@1e-4": cg4.sum().item() / float(v),
                #"avgcounts@1e-3": cg3.sum().item() / float(v),
                "avgcounts@1e-2": cg2.sum().item() / float(v),
                #"maxcounts@1e-4": cg4.sum(0).max().item() / float(v),
                #"maxcounts@1e-3": cg3.sum(0).max().item() / float(v),
                "maxcounts@1e-2": cg2.sum(0).max().item(),
                #"mincounts@1e-4": cg4.sum(0).min().item() / float(v),
                #"mincounts@1e-3": cg3.sum(0).min().item() / float(v),
                "mincounts@1e-2": cg2.sum(0).min().item(),
                "maxcounts": counts.sum(0).max().item(),
                "mincounts": counts.sum(0).min().item(),
            }, step=WANDB_STEP)
            del cg2
            del counts

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()

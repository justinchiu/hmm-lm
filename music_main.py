
import time as timep

import sys

import math
import time

from pathlib import Path

import numpy as np

import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_

import torch_struct as ts

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import torchtext
import datasets.music as music
from datasets.data import MusicIterator

from args import get_args

from utils import set_seed, get_config, get_name, get_mask_lengths
from utils import Pack
from utils import plot_counts, print_gpu_mem

import wandb

#th.autograd.set_detect_anomaly(True)

valid_schedules = ["reducelronplateau"]

WANDB_STEP = -1

BEST_VALID = -math.inf
PREV_SAVE = None

V = 88

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
        # do not save on dryruns
        if not wandb.run._settings._offline:
            save_f = f"wandb_checkpoints/{name}/{WANDB_STEP}_{-valid_losses.evidence / valid_n:.3f}.pth"
            print(f"Saving model to {save_f}")
            Path(save_f).parent.mkdir(parents=True, exist_ok=True)
            th.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": model.config,
            }, save_f)
            if PREV_SAVE is not None and PREV_SAVE != save_f:
                # dont remove overwritten file
                Path(PREV_SAVE).unlink()
            PREV_SAVE = save_f

        BEST_VALID = valid_losses.evidence
        wandb.run.summary["best_valid_ppl"] = math.exp(-BEST_VALID / valid_n)
        wandb.run.summary["best_valid_loss"] = BEST_VALID / valid_n


def report(losses, n, prefix, start_time=None):
    loss = losses.evidence
    elbo = losses.elbo
    # cap loss otherwise overflow
    #loss = loss if loss > -1e7 else -1e7
    str_list = [
        f"{prefix}: log_prob = {loss:.2f}",
        f"xent(word) = {-loss / n:.2f}",
        #f"ppl = {math.exp(-loss / n):.2f}",
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

            text, mask, lengths = batch
            n_tokens = mask.sum()

            if args.iterator != "bptt":
                lpz, last_states = None, None

            losses, lpz, _ = model.score(
                text,
                lpz=lpz, last_states = last_states,
                mask=mask, lengths=lengths,
            )
            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n

def cached_eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    with th.no_grad():
        model.train(False)
        lpz = None
        start, transition, emission = model.compute_parameters(model.word2state)
        if not model.eff:
            # assert that transition and emission are well-formed
            bigt = transition.logsumexp(-1).abs().max()
            assert bigt < 1e-4, f"{bigt}"
            bige = emission.logsumexp(-1).abs().max()
            assert bige < 1e-4, f"{bige}"
            # log entropy of transition and emission

            He = -(emission.exp() * emission).sum()
            Ht = -(transition.exp() * transition).sum()
            print(f"Total transition entropy {Ht:.2f} || Total emission entropy {He.sum():.2f}")

        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text = batch.text

            text, mask, lengths = batch
            n_tokens = mask.sum()
            N, T = text.shape

            if lpz is not None and args.iterator == "bptt":
                start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)

            log_potentials = model.clamp(
                text, start, transition, emission, word2state,
            )
            losses, lpz = model.compute_loss(log_potentials, mask, lengths)

            if word2state is not None:
                idx = th.arange(N, device=model.device)
                last_words = text[idx, lengths-1]
                last_states = model.word2state[last_words]

            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens
    return Pack(evidence = total_ll, elbo = total_elbo), n

def fast_eval_loop(
    args, V, iter, model,
    do_svd = False,
):
    total_ll = 0
    total_elbo = 0
    n = 0

    with th.no_grad():
        model.train(False)
        lpz = None
        start, transition, emission = model.compute_parameters(model.word2state)

        if hasattr(model, "temp"):
            print(f"Model temp {model.temp.item():.2f}")

        # entropy
        # assert that transition and emission are well-formed
        myt = model.transition(print_max=True)
        #myt = model.transition()
        bigt = myt.logsumexp(-1).abs().max()
        assert bigt < 1e-4, f"{bigt}"
        bige = emission.logsumexp(-1).abs().max()
        assert bige < 1e-4, f"{bige}"
        # log entropy of transition and emission

        He = -(emission.exp() * emission).sum()
        Ht = -(myt.exp() * myt).sum()
        print(f"Total transition entropy {Ht:.2f} || Total emission entropy {He.sum():.2f}")

        if do_svd:
            # svd
            _,s,_ = myt.exp().svd(compute_uv=False)
            data = [[i,v] for i,v in enumerate(s.cpu().detach().numpy())]
            table = wandb.Table(data=data, columns = ["index", "value"]) 
            wandb.log({
                "transition_entropy": Ht,
                "emission_entropy": He,
                "svd": wandb.plot.line(table, "index", "value", title="Singular Values"),
            }, step=WANDB_STEP)


        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text, mask, lengths = batch
            n_tokens = mask.sum()
            N, T, num_notes = text.shape

            if lpz is not None and args.iterator == "bptt":
                #start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)
                raise NotImplementedError()

            losses, lpz = model.compute_loss(
                text, start, transition, emission, word2state,
                mask=mask, lengths=lengths)

            if word2state is not None:
                idx = th.arange(N, device=model.device)
                last_words = text[idx, lengths-1]
                last_states = model.word2state[last_words]

            total_ll += losses.evidence.detach()
            if losses.elbo is not None:
                total_elbo += losses.elbo.detach()
            n += n_tokens

    return Pack(evidence = total_ll, elbo = total_elbo), n

def collect_counts_loop(
    args, V, iter, model,
):
    from models.analysis_utils import CountCollector
    count_collector = CountCollector(model)

    model.train(False)
    lpz = None

    word2state = model.word2state
    for i, batch in enumerate(iter):
        if hasattr(model, "noise_scale"):
            model.noise_scale = 0

        text = batch.text

        text, mask, lengths = batch
        n_tokens = mask.sum()
        N, T = text.shape

        if lpz is not None and args.iterator == "bptt":
            #start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)
            raise NotImplementedError()

        if word2state is not None:
            idx = th.arange(N, device=model.device)
            last_words = text[idx, lengths-1]
            last_states = model.word2state[last_words]

        count_collector.collect_counts(text, mask, lengths)
    return count_collector

def mixed_cached_eval_loop(
    args, V, iter, model,
):
    total_ll = 0
    total_elbo = 0
    n = 0
    with th.no_grad():
        model.train(False)
        lpz = None

        start = model.start().cpu()
        emission = model.mask_emission(model.emission_logits(), model.word2state).cpu()

        # blocked transition
        num_blocks = 128
        block_size = model.C // num_blocks
        next_state_proj = (model.next_state_proj.weight
            if hasattr(model, "next_state_proj")
            else model.next_state_emb()
        )
        transition = th.empty(model.C, model.C, device=th.device("cpu"), dtype=emission.dtype)
        for s in range(0, model.C, block_size):
            states = range(s, s+block_size)
            x = model.trans_mlp(model.dropout(
                model.state_emb.weight[states]
                if hasattr(model.state_emb, "weight")
                else model.state_emb(th.LongTensor(states).to(model.device))
            ))
            y = (x @ next_state_proj.t()).log_softmax(-1)
            transition[states] = y.to(transition.device)

        #start0, transition0, emission0 = model.compute_parameters(model.word2state)
        # th.allclose(transition, transition0)
        word2state = model.word2state
        for i, batch in enumerate(iter):
            if hasattr(model, "noise_scale"):
                model.noise_scale = 0

            text = batch.text

            text, mask, lengths = batch
            n_tokens = mask.sum()
            N, T = text.shape

            if lpz is not None and args.iterator == "bptt":
                # hopefully this isn't too slow on cpu
                start = (lpz[:,:,None] + transition[last_states,:]).logsumexp(1)

            log_potentials = model.clamp(
                text, start, transition, emission, word2state
            ).to(model.device)

            losses, lpz = model.compute_loss(log_potentials, mask, lengths)
            lpz = lpz.cpu()

            idx = th.arange(N, device=model.device)
            last_words = text[idx, lengths-1]
            last_states = model.word2state[last_words]

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

            if args.iterator == "bucket":
                lpz = None
                last_states = None
            #print(" ".join([model.V.itos[x] for x in text[0].tolist()]))

            # set noise scale
            if hasattr(model, "noise_scale"):
                noise_scale = noise_scales[
                    min(WANDB_STEP, args.noise_anneal_steps-1)
                ] if args.noise_anneal_steps > 0 else model.init_noise_scale
                model.noise_scale = noise_scale
                wandb.log({
                    "noise_scale": noise_scale,
                }, step=WANDB_STEP)

            text, mask, lengths = batch
            n_tokens = mask.sum()
            N, T, num_notes = text.shape

            if model.timing:
                start_forward = timep.time()

            # check if iterator == bptt

            if hasattr(args, "eff") and args.eff:
                losses, _, _= model.score_rff(
                    text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)
                """
                losses2, _, _ = model.score(
                    text, lpz=lpz, last_states=last_states, mask=mask, lengths=lengths)
                import pdb; pdb.set_trace()
                """
            else:
                losses, lpz, last_states = model.score(
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

            #print("stateemb")
            #print(model.state_emb.grad.max().item(), model.state_emb.grad.min().item())
            #print(model.next_state_emb.grad.max().item(), model.next_state_emb.grad.min().item())
            #print("colbanded")
            #print(model.col_banded_transition.grad.max().item(), model.col_banded_transition.grad.min().item())
            #import pdb; pdb.set_trace()

            if model.timing:
                print(f"backward time: {timep.time() - start_backward}")
            gradnorm = clip_grad_norm_(parameters, args.clip)
            if args.schedule not in valid_schedules and scheduler is not None:
                # sched before opt since we want step = 1?
                # this is how huggingface does it
                scheduler.step()
            optimizer.step()
            wandb.log({
                "running_training_loss": total_ll / n,
                "running_training_ppl": math.exp(min(-total_ll / n, 700)),
                "running_training_elbo": total_elbo / n,
                "gradnorm": gradnorm,
            }, step=WANDB_STEP)
            if model.timing:
                print_gpu_mem()

            if model.timing:
                print(f"gradnorm {i}: {gradnorm} || sur {loss} || ev {losses.evidence}")
                import pdb; pdb.set_trace()

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
                elif args.model == "hmm" or args.model == "lhmm" or args.model == "sparsekernelhmm":
                    eval_fn = cached_eval_loop
                elif args.model == "blhmm" or args.model == "sblhmm" or args.model == "bandedhmm":
                    eval_fn = fast_eval_loop
                else:
                    eval_fn = eval_loop
                valid_losses, valid_n  = eval_fn(
                    args, V, valid_iter, model,
                )
                #import pdb; pdb.set_trace()
                report(valid_losses, valid_n, "Valid eval", v_start_time)
                wandb.log({
                    "valid_loss": valid_losses.evidence / valid_n,
                    "valid_ppl": math.exp(-valid_losses.evidence / valid_n),
                }, step=WANDB_STEP)

                update_best_valid(
                    valid_losses, valid_n, model, optimizer, scheduler, args.name)

                wandb.log({
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=WANDB_STEP)
                if scheduler is not None:
                    scheduler.step(valid_losses.evidence)

                # remove this later?
                if args.log_counts > 0 and args.keep_counts > 0:
                    # TODO: FACTOR OUT
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

    return Pack(evidence = total_ll, elbo = total_elbo), n


def main():
    global WANDB_STEP
    #args = get_args(argstrings.lm_args)
    args = get_args()

    set_seed(args.seed)

    device = th.device("cpu" if args.devid < 0 else f"cuda:{args.devid}")
    args.device = device
    aux_device = th.device("cpu" if args.aux_devid < 0 else f"cuda:{args.aux_devid}")
    args.aux_device = aux_device

    if args.dbg_double:
        # test higher precision
        th.set_default_tensor_type(th.cuda.DoubleTensor)
        # reminder: never use 'as th' again

    """
    for data in [music.JSB_CHORALES, music.MUSE_DATA, music.NOTTINGHAM, music.PIANO_MIDI]:
        d = music.load_data(data)
        d_iter = MusicIterator(
            dataset = (d["train"]["sequences"], d["train"]["sequence_lengths"]),
            batch_size = 128,
            sort_key = lambda x: x[1].sum(),
        )
        for b,m,l in d_iter:
            print(b.shape)
    """
    data_map = {
        "jsb": music.JSB_CHORALES,
        "muse": music.MUSE_DATA,
        "nottingham": music.NOTTINGHAM,
        "piano": music.PIANO_MIDI,
    }
    data = music.load_data(data_map[args.music_dataset])
    def get_iter(data, split, bsz):
        return MusicIterator(
            dataset = (data[split]["sequences"], data[split]["sequence_lengths"]),
            batch_size = bsz,
            sort_key = lambda x: x[1].sum(),
            device = device,
        )
    train_iter = get_iter(data, "train", args.bsz)
    valid_iter = get_iter(data, "valid", args.eval_bsz)
    test_iter = get_iter(data, "test", args.eval_bsz)

    if args.no_shuffle_train:
        train_iter.shuffle = False
    #import pdb; pdb.set_trace()
    """
    args = get_config(args.model_config, device)
    config = Pack(args.items() | args.items())
    # for now, until migrate to argparse
    config.model = args.type
    """
    name = get_name(args, music=True)
    import tempfile
    wandb.init(project="music-hmm", name=name, config=args, dir=tempfile.mkdtemp())
    args.name = name

    print(" ".join(sys.argv))
    print(args)

    model = None
    if args.model == "lstm":
        model = LstmLm(V, args)
    elif args.model == "hmm":
        from models.hmmlm import HmmLm
        model = HmmLm(V, args)
    elif args.model == "lhmm":
        from models.lhmmlm import LHmmLm
        model = LHmmLm(V, args)
    elif args.model == "blhmm":
        from music_models.blhmmlm import BLHmmLm
        model = BLHmmLm(V, args)
    elif args.model == "bandedhmm":
        from music_models.banded_hmmlm import BandedHmmLm
        model = BandedHmmLm(V, args)
    elif args.model == "sblhmm":
        from models.sblhmmlm import SblHmmLm
        model = SblHmmLm(V, args)
        # single projection doesnt work due to rank issues, try multiple projection
        #from models.msblhmmlm import MsblHmmLm
        #model = MsblHmmLm(V, args)
    elif args.model == "sparsekernelhmm":
        from models.sparse_kernel_hmmlm import SparseKernelHmmLm
        model = SparseKernelHmmLm(V, args)
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
    wandb.run.summary["num_params"] = num_params

    # load frozen transition
    if args.frozen_pretrained_transition is not None:
        past_chp = th.load(args.frozen_pretrained_transition)
        from models.sblhmmlm import SblHmmLm
        past_model = SblHmmLm(V, past_chp["args"])
        past_model.load_state_dict(past_chp["model"])
        transition = past_model.transition().detach().clone()
        model.set_frozen_transition(transition)
        # cleanup
        del past_model
        del past_chp

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
        # make sure this is uncommented
        if args.eval_only != "none":
            model.load_state_dict(th.load(args.eval_only)["model"])
        from utils import dump_transition
        dump_transition(model)

        v_start_time = time.time()
        #valid_losses, valid_n = eval_loop(
        #valid_losses, valid_n = cached_eval_loop(
        if args.model == "mshmm" or args.model == "factoredhmm":
            if args.num_classes > 2 ** 15:
                eval_fn = mixed_cached_eval_loop
            else:
                eval_fn = cached_eval_loop
        elif args.model == "hmm" or args.model =="lhmm":
            eval_fn = cached_eval_loop
        elif args.model == "blhmm" or args.model == "sblhmm" or args.model == "bandedhmm":
            eval_fn = fast_eval_loop
        else:
            eval_fn = eval_loop
        #eval_fn = cached_eval_loop if args.model == "mshmm" else eval_loop
        valid_losses, valid_n = eval_fn(
            args, V, valid_iter, model,
        )
        report(valid_losses, valid_n, f"Valid perf", v_start_time)

        # count states
        valid_counts = collect_counts_loop(
            args, V, valid_iter, model,
        )
        valid_counts.print_counts()

        t_start_time = time.time()
        test_losses, test_n = eval_fn(
            args, V, test_iter, model,
        )
        report(test_losses, test_n, f"Test perf", t_start_time)

        # count states
        #test_counts = collect_counts_loop(
            #args, V, test_iter, model,
        #)
        #test_counts.print_counts()

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
    elif args.schedule == "none":
        scheduler = None
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
        elif args.model == "blhmm" or args.model == "sblhmm" or args.model == "bandedhmm":
            eval_fn = fast_eval_loop
        else:
            eval_fn = eval_loop
        valid_losses, valid_n  = eval_fn(args, V, valid_iter, model)
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

    # won't use best model. Rerun with eval_only
    t_start_time = time.time()
    test_losses, test_n = eval_fn(
        args, V, test_iter, model,
    )
    report(test_losses, test_n, f"Test perf", t_start_time)

if __name__ == "__main__":
    print(" ".join(sys.argv))
    main()
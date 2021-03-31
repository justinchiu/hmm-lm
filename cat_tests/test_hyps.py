
import sys

from tqdm import trange
from itertools import zip_longest

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from models.linear_utils import get_2d_array, project_logits

class Cat(nn.Module):
    def __init__(
        self,
        num_starts,
        num_classes,
        emb_dim,
        feature_dim,
        temp = 1.,
        xavier_init=True,
        sm=False,
        l2norm=False,
        random_feature=False,
        learn_temp=False,
    ):
        torch.manual_seed(0)
        super(Cat, self).__init__()
        self.sm = sm
        self.l2norm = l2norm
        self.random_feature = random_feature

        self.temp = nn.Parameter(torch.FloatTensor([temp]))
        if not learn_temp:
           self.temp.requires_grad = False

        self.start_emb = nn.Parameter(
            torch.randn(num_starts, emb_dim),
        )
        self.output_emb = nn.Parameter(
            torch.randn(num_classes, emb_dim),
        )

        # init
        if xavier_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # leave this at orthogonal init
        self.proj_shape = (feature_dim, emb_dim)
        self.proj = nn.Parameter(get_2d_array(*self.proj_shape).T)
        if self.random_feature:
            self.proj.requires_grad = False
            self.counter = 1

    def sample_proj(self):
        if (self.counter % 100) == 0:
            self.proj.copy_(get_2d_array(*self.proj_shape).T)
        self.counter += 1
        return self.proj

    def log_probs(self):
        return self.logits().log_softmax(-1)

    def logits(self):
        fx = self.start_emb
        fy = self.output_emb
        if self.l2norm:
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)
        if self.sm:
            return (fx @ fy.T) / self.temp
        else:
            proj = (self.proj if not self.random_feature
                else self.sample_proj().to(self.proj.device))
            L = fx @ proj
            R = fy @ proj
            #L = fx @ proj - fx.square().sum(-1, keepdim=True) / 2
            #R = fy @ proj - fy.square().sum(-1, keepdim=True) / 2
            return (L[:,None,:] + R[None,:,:]).logsumexp(-1) / self.temp

    def kl(self, true_dist):
        return (true_dist.exp() * (true_dist - self.log_probs())).sum(-1).mean()

def H(lp):
    return -(lp.exp() * lp).sum(-1)

emb_dim = 128
feature_dim_ratio_grid = [1, 2, 4, 8, 16]
feature_dim_ratio_grid = [0.5, 1, 2, 4, 8]
num_classes_grid = [1024, 2048]

num_classes_grid = [128, 256, 512]
num_classes_grid = [128]

device = torch.device("cuda:0")
num_steps = 20000
#num_steps = 4000
#num_steps = 2000

def init_optimizer(model):
    parameters = list(model.parameters())
    return AdamW(
        parameters,
        lr = 1e-3,
        #lr = 1e-4,
        betas = (0.9, 0.999),
        weight_decay = 0.,
    )

def train(true_dist, model, num_steps, check_svs=0):
    optimizer = init_optimizer(model)
    kls = []
    svs = []
    #for i in trange(num_steps):
    for i in range(num_steps):
        optimizer.zero_grad()
        kl = model.kl(true_dist)
        kls.append(kl.item())
        kl.backward()
        optimizer.step()
        if check_svs:
            do_check = (i + 1) % (num_steps // check_svs) == 0
            if do_check:
                logits = model.logits()
                lp = logits.log_softmax(-1)
                u,s,v = lp.exp().svd()
                svs.append(s)
    #return kls if check_svs == 0 else (kls, svs)
    return kls, svs

def print_stats(model):
    logits = model.logits()
    lp = logits.log_softmax(-1)
    u,s,v = lp.exp().svd()
    #num_sv = (s > 1e-5).sum().item()
    num_sv = (s > 1).sum().item()
    minproj = f"{model.proj.min().item() if not model.sm else 0:.2f}"
    maxproj = f"{model.proj.max().item() if not model.sm else 0:.2f}"

    minsemb = f"{model.start_emb.min().item():.2f}"
    maxsemb = f"{model.start_emb.max().item():.2f}"
    minoemb = f"{model.output_emb.min().item():.2f}"
    maxoemb = f"{model.output_emb.max().item():.2f}"
    minemb = min(minsemb,minoemb)
    maxemb = max(maxsemb,maxoemb)

    print(f"num sv > 1: {num_sv} || H: {H(lp).mean().item():.2f} || min/max logit: {logits.min().item():.2f}/{logits.max().item():.2f} || proj: {minproj}/{maxproj} || emb: {minemb}/{maxemb} || temp {model.temp.item():.2f}")
    return s

def plot(losses, svs, name, num_starts, num_classes, num_features=0, learn_temp=False):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=1.5)
    fig, ax = plt.subplots()
    g = sns.lineplot(x=np.arange(len(losses)), y=losses, ax=ax)
    fig.savefig(f"cat_tests/plots/{name}-{num_starts}-{num_classes}-{'lt' if learn_temp else 'nol'}.png")
    plt.close(fig)

    sns.set(font_scale=1.5)
    fig, ax = plt.subplots()
    g = sns.scatterplot(x=np.arange(len(svs)),y=svs.cpu().detach().numpy(), ax=ax)
    if num_features > 0:
        fig.savefig(f"cat_tests/plots/svs-{name}-{num_starts}-{num_classes}-{num_features}-{'lt' if learn_temp else 'nol'}.png")
    else:
        fig.savefig(f"cat_tests/plots/svs-{name}-{num_starts}-{num_classes}-{'lt' if learn_temp else 'nol'}.png")
    plt.close(fig)

def plot_svs(svs_list, name, num_starts, num_classes, num_features=0, learn_temp=False):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=1.5)

    fig, axes = plt.subplots(ncols=len(svs_list), sharey=True)
    if len(svs_list) == 1:
        # wrap singleton list?
        axes = [axes]
    for ax, svs in zip(axes, svs_list):
        g = sns.scatterplot(x=np.arange(len(svs)),y=svs.cpu().detach().numpy(), ax=ax)
    if num_features > 0:
        fig.savefig(f"cat_tests/trainsv_plots/trainsvs-{name}-{num_starts}-{num_classes}-{num_features}-{'lt' if learn_temp else 'nol'}.png")
    else:
        fig.savefig(f"cat_tests/trainsv_plots/trainsvs-{name}-{num_starts}-{num_classes}-{'lt' if learn_temp else 'nol'}.png")
    plt.close(fig)

def run_fit(
    true_dist_fn,
    num_classes,
    feature_dim_ratio = None,
    feature_dim = None,
    emb_dim = 128,
    random_feature=False,
    learn_temp=False,
    plot_losses = False,
    check_svs = 0,
):
    true_dist = true_dist_fn(num_classes)
    num_starts = true_dist.shape[0]

    # softmax
    model = Cat(
        num_starts,
        num_classes, emb_dim, feature_dim=1,
        sm=True,
        learn_temp = learn_temp,
    )
    model.to(device)
    losses, svs_train = train(true_dist, model, num_steps, check_svs)
    print(f"SM queries {num_starts} keys {num_classes} edim {emb_dim} ||| KL {losses[-1]:.4} <<<")
    svs = print_stats(model)
    sm_losses = losses

    if plot_losses:
        plot(losses, svs, "sm", num_starts, num_classes, learn_temp=learn_temp)
    if check_svs != 0:
        plot_svs(svs_train, "sm", num_starts, num_classes, learn_temp=learn_temp)

    # kernel
    if feature_dim is None:
        feature_dim = int(num_classes // feature_dim_ratio)
    model = Cat(
        num_starts,
        num_classes, emb_dim, feature_dim,
        random_feature = random_feature,
        learn_temp = learn_temp,
    )
    model.to(device)
    losses, svs_train = train(true_dist, model, num_steps, check_svs)
    print(f"K queries {num_starts} keys {num_classes} feats {feature_dim} edim {emb_dim} ||| KL: {losses[-1]:.4f} <<<")
    svs = print_stats(model)
    k_losses = losses

    if plot_losses:
        plot(losses, svs, "k", num_starts, num_classes, feature_dim, learn_temp)
    if check_svs != 0:
        plot_svs(svs_train, "k", num_starts, num_classes, feature_dim, learn_temp)

    return sm_losses, k_losses

    """
    # l2norm
    model = Cat(
        num_starts,
        num_classes, emb_dim, feature_dim, l2norm=True)
    model.to(device)
    losses = train(true_dist, model, num_steps)
    print(num_starts, num_classes, feature_dim, "l2norm", losses[-1])
    """
PLOT = False
PLOT = True
if PLOT:
    print("Plotting losses")
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_classes,
            num_classes, 128, 1,
            temp=1, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        print_stats(true_model)
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes = 128,
        feature_dim = 64,
        emb_dim = 128,
        plot_losses = True,
        check_svs = 4,
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes = 128,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
        plot_losses = True,
        check_svs = 4,
    )
    print()

    """
    print("Plotting low queries high keys")
    for num_starts in [32, 64]:
        def true_dist_sm(num_classes):
            true_model = Cat(
                num_starts,
                num_classes, 128, 1,
                temp=1, xavier_init=False, sm=True)
            true_model.to(device)
            true_dist = true_model.log_probs().detach()
            print(f"True dist H: {H(true_dist).mean().item():.2f}")
            print_stats(true_model)
            return true_dist
        run_fit(
            true_dist_sm,
            num_classes = 1024,
            feature_dim = 64,
            emb_dim = 128,
            plot_losses = True,
            check_svs = 4,
        )
        print("Learn temp")
        run_fit(
            true_dist_sm,
            num_classes = 1024,
            feature_dim = 64,
            emb_dim = 128,
            learn_temp = True,
            plot_losses = True,
            check_svs = 4,
        )
        print()
    """
    
    print("Smoothed One-hot True Dist")
    eps = 1e-4
    # true dist is one-hot + smoothing
    def true_dist_onehot(num_classes):
        logits = (torch.eye(num_classes, device=device) + eps).log()
        true_dist = logits.log_softmax(-1)
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        u,s,v = true_dist.exp().svd()
        plot_svs([s], "trueeye", num_classes, num_classes)
        return true_dist
    run_fit(
        true_dist_onehot,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
        plot_losses = True,
        check_svs = 4,
    )
    print("Learn temp")
    run_fit(
        true_dist_onehot,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
        plot_losses = True,
        check_svs = 4,
    )
    print()

temp_grid = [1, 2, 3]
print("Higher entropy is easier to fit")
for temp in temp_grid:
    print(f"Temperature {temp}")
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_classes,
            num_classes, 128, 1,
            temp=temp, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        print_stats(true_model)
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
    )
    print()

print("Higher rank is harder to fit")
for emb_dim in [32, 64, 128, 256, 512]:
    print(f"True model emb_dim: {emb_dim}")
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_classes,
            num_classes, emb_dim, 1,
            temp=1, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        print_stats(true_model)
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
    )
print()

print("Higher number of classes (keys) is harder to fit")
def true_dist_sm(num_classes):
    true_model = Cat(
        128,
        num_classes, 128, 1,
        temp=1, xavier_init=False, sm=True)
    true_model.to(device)
    true_dist = true_model.log_probs().detach()
    print(f"True dist H: {H(true_dist).mean().item():.2f}")
    print_stats(true_model)
    return true_dist
num_classes_grid = [64, 128, 256, 512, 1024]
for num_classes in num_classes_grid:
    run_fit(
        true_dist_sm,
        num_classes = num_classes,
        feature_dim = 64,
        emb_dim = 128,
    )
print("Learn temp")
for num_classes in num_classes_grid:
    run_fit(
        true_dist_sm,
        num_classes = num_classes,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
    )
print()

num_starts_grid = [64, 128, 256, 512, 1024]
print("Higher number of starts (queries) is harder to fit")
for num_starts in num_starts_grid:
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_starts,
            num_classes, 128, 1,
            temp=1, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        print_stats(true_model)
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes = 256,
        feature_dim = 64,
        emb_dim = 128,
        learn_temp = True,
    )
    print()

"""
print("Higher entropy is easier to fit")
print("Rows Cols {Feats} KL")
for eps in [1e-4, 1e-3, 1e-2]:
    print(f"Smoothing eps: {eps}")
    print()
"""

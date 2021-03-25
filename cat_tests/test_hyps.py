
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
            return (L[:,None,:] + R[None,:,:]).logsumexp(-1) / self.temp

    def kl(self, true_dist):
        return (true_dist.exp() * (true_dist - self.log_probs())).sum(-1).mean()

def H(lp):
    return -(lp.exp() * lp).sum(-1)

emb_dim = 32
feature_dim_ratio_grid = [1, 2, 4, 8, 16]
feature_dim_ratio_grid = [0.5, 1, 2, 4, 8]
num_classes_grid = [1024, 2048]

num_classes_grid = [128, 256, 512]
num_classes_grid = [128]

device = torch.device("cuda:0")
num_steps = 2000
#num_steps = 4000

def init_optimizer(model):
    parameters = list(model.parameters())
    return AdamW(
        parameters,
        lr = 1e-3,
        #lr = 1e-4,
        betas = (0.9, 0.999),
        weight_decay = 0.,
    )

def train(true_dist, model, num_steps):
    optimizer = init_optimizer(model)
    kls = []
    #for i in trange(num_steps):
    for i in range(num_steps):
        optimizer.zero_grad()
        kl = model.kl(true_dist)
        kls.append(kl.item())
        kl.backward()
        optimizer.step()
    return kls

def print_stats(model):
    logits = model.logits()
    lp = logits.log_softmax(-1)
    u,s,v = lp.exp().svd()
    #num_sv = (s > 1e-5).sum().item()
    num_sv = (s > 1).sum().item()
    print(f"num sv > 1: {num_sv} || H: {H(lp).mean().item():.2f} || min/max logit: {logits.min().item():.2f}/{logits.max().item():.2f} || temp {model.temp.item():.2f}")

def plot(losses, name, num_starts, num_classes, learn_temp=False):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=1.5)
    g = sns.lineplot(x=np.arange(len(losses)), y=losses)
    plt.savefig(f"{name}-{num_starts}-{num_classes}-{'lt' if learn_temp else 'nol'}.png")

def run_fit(
    true_dist_fn,
    num_classes_grid,
    feature_dim_ratio_grid=[],
    feature_dim_grid=[],
    random_feature=False,
    learn_temp=False,
    plot_losses = False,
):
    for num_classes in num_classes_grid:
        true_dist = true_dist_fn(num_classes)
        num_starts = true_dist.shape[0]

        # softmax
        model = Cat(
            num_starts,
            num_classes, emb_dim, feature_dim=1,
            sm=True,
        )
        model.to(device)
        model.to(device)
        losses = train(true_dist, model, num_steps)
        print("SM", num_starts, num_classes, f"||| {losses[-1]:.2f} <<<")
        print_stats(model)

        if plot_losses:
            plot(losses, "sm", num_starts, num_classes, False)

        # kernel
        for feature_dim_ratio, feature_dim in zip_longest(
            feature_dim_ratio_grid, feature_dim_grid
        ):
            if feature_dim is None:
                feature_dim = int(num_classes // feature_dim_ratio)
            model = Cat(
                num_starts,
                num_classes, emb_dim, feature_dim,
                random_feature = random_feature,
                learn_temp = learn_temp,
            )
            model.to(device)
            losses = train(true_dist, model, num_steps)
            print("K", num_starts, num_classes, f"||| {losses[-1]:.2f} <<<")
            print_stats(model)

            if plot_losses:
                plot(losses, "k", num_starts, num_classes, learn_temp)

            """
            # l2norm
            model = Cat(
                num_starts,
                num_classes, emb_dim, feature_dim, l2norm=True)
            model.to(device)
            losses = train(true_dist, model, num_steps)
            print(num_starts, num_classes, feature_dim, "l2norm", losses[-1])
            """
"""
print("Higher entropy is easier to fit")
print("Rows Cols {Feats} KL")
for eps in [1e-4, 1e-3, 1e-2]:
    print(f"Smoothing eps: {eps}")
    print("Smoothed One-hot True Dist")
    # true dist is one-hot + smoothing
    def true_dist_onehot(num_classes):
        logits = (torch.eye(num_classes, device=device) + eps).log()
        true_dist = logits.log_softmax(-1)
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        return true_dist
    run_fit(true_dist_onehot)
    print()
"""

print("Plotting losses")
def true_dist_sm(num_classes):
    true_model = Cat(
        num_classes,
        num_classes, emb_dim, 1,
        temp=1, xavier_init=False, sm=True)
    true_model.to(device)
    true_dist = true_model.log_probs().detach()
    print(f"True dist H: {H(true_dist).mean().item():.2f}")
    return true_dist
run_fit(
    true_dist_sm,
    num_classes_grid = [128],
    feature_dim_grid = [64],
    plot_losses = True,
)
print("Learn temp")
run_fit(
    true_dist_sm,
    num_classes_grid = [128],
    feature_dim_grid = [64],
    learn_temp = True,
    plot_losses = True,
)

temp_grid = [1, 2, 3, 4, 5]
print("Lower entropy is harder to fit")
for temp in temp_grid:
    print(f"Temperature {temp}")
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_classes,
            num_classes, emb_dim, 1,
            temp=temp, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes_grid = [128],
        feature_dim_grid = [64, 128, 256, 512],
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes_grid = [128],
        feature_dim_grid = [64, 128, 256, 512],
        learn_temp = True,
    )
    print()

print("Higher rank is harder to fit")
def true_dist_sm(num_classes):
    true_model = Cat(
        num_classes,
        num_classes, emb_dim, 1,
        temp=1, xavier_init=False, sm=True)
    true_model.to(device)
    true_dist = true_model.log_probs().detach()
    print(f"True dist H: {H(true_dist).mean().item():.2f}")
    return true_dist
run_fit(
    true_dist_sm,
    num_classes_grid = [64, 128, 256],
    feature_dim_grid = [64, 128, 256, 512],
)
print("Learn temp")
run_fit(
    true_dist_sm,
    num_classes_grid = [64, 128, 256],
    feature_dim_grid = [64, 128, 256, 512],
    learn_temp = True,
)
print()

print("Higher number of classes (keys) is harder to fit")
def true_dist_sm(num_classes):
    true_model = Cat(
        128,
        num_classes, emb_dim, 1,
        temp=1, xavier_init=False, sm=True)
    true_model.to(device)
    true_dist = true_model.log_probs().detach()
    print(f"True dist H: {H(true_dist).mean().item():.2f}")
    return true_dist
run_fit(
    true_dist_sm,
    num_classes_grid = [128, 256, 512, 1024, 2048],
    feature_dim_grid = [64, 128, 256, 512],
)
print("Learn temp")
run_fit(
    true_dist_sm,
    num_classes_grid = [128, 256, 512, 1024, 2048],
    feature_dim_grid = [64, 128, 256, 512],
    learn_temp = True,
)
print()

num_starts_grid = [32, 64, 128, 256]
print("Lower number of starts (queries) is easier to fit")
for num_starts in num_starts_grid:
    def true_dist_sm(num_classes):
        true_model = Cat(
            num_starts,
            num_classes, emb_dim, 1,
            temp=1, xavier_init=False, sm=True)
        true_model.to(device)
        true_dist = true_model.log_probs().detach()
        print(f"True dist H: {H(true_dist).mean().item():.2f}")
        return true_dist
    run_fit(
        true_dist_sm,
        num_classes_grid = [1024],
        feature_dim_grid = [64, 128, 256, 512],
    )
    print("Learn temp")
    run_fit(
        true_dist_sm,
        num_classes_grid = [1024],
        feature_dim_grid = [64, 128, 256, 512],
        learn_temp = True,
    )
    print()

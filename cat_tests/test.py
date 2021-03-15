
from tqdm import trange

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
    ):
        torch.manual_seed(0)
        super(Cat, self).__init__()
        self.sm = sm
        self.l2norm = l2norm
        self.temp = temp

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
        self.proj = nn.Parameter(
            get_2d_array(feature_dim, emb_dim).T
        )

    def log_probs(self):
        return self.sm_log_probs() if self.sm else self.k_log_probs()

    def k_log_probs(self):
        fx = self.start_emb
        fy = self.output_emb
        if self.l2norm:
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)
        logits = project_logits(
            fx[None],
            fy[None],
            self.proj,
            rff_method = "log",
        )[0]
        return logits.log_softmax(-1)

    def sm_log_probs(self):
        fx = self.start_emb
        fy = self.output_emb
        if self.l2norm:
            fx = fx / fx.norm(dim=-1, keepdim=True)
            fy = fy / fy.norm(dim=-1, keepdim=True)
        return ((fx @ fy.T) / self.temp).log_softmax(-1)

    def kl(self, true_dist):
        return (true_dist.exp() * (true_dist - self.log_probs())).sum(-1).mean()

    def H(self):
        lp = self.log_probs()
        return -(lp.exp() * lp).sum(-1)

emb_dim = 256
num_starts = 128
feature_dim_ratio_grid = [2, 4, 8,]
num_classes_grid = [1024, 2048, 4096, 8192]
num_classes_grid = [1024, 2048]

device = torch.device("cuda:0")
num_steps = 10000

def init_optimizer(model):
    parameters = list(model.parameters())
    return AdamW(
        parameters,
        lr = 1e-3,
        betas = (0.9, 0.999),
        weight_decay = 0.,
    )

def train(true_dist, model, num_steps):
    optimizer = init_optimizer(model)
    kls = []
    for i in trange(num_steps):
        optimizer.zero_grad()
        kl = model.kl(true_dist)
        kls.append(kl.item())
        kl.backward()
        optimizer.step()
    return kls


for num_classes in num_classes_grid:
    true_model = Cat(
        num_starts,
        num_classes, emb_dim, 1,
        temp=1, xavier_init=False, sm=True)
    true_model.to(device)
    true_dist = true_model.log_probs().detach()
    true_H = true_model.H()

    # softmax
    model = Cat(
        num_starts,
        num_classes, emb_dim, feature_dim=1,
        sm=True,
    )
    model.to(device)
    model.to(device)
    losses = train(true_dist, model, num_steps)
    print("SM", num_starts, num_classes, losses[-1])

    # kernel
    for feature_dim_ratio in feature_dim_ratio_grid:
        feature_dim = num_classes // feature_dim_ratio
        model = Cat(
            num_starts,
            num_classes, emb_dim, feature_dim)
        model.to(device)
        losses = train(true_dist, model, num_steps)
        print(num_starts, num_classes, feature_dim, losses[-1])

        # l2norm
        model = Cat(
            num_starts,
            num_classes, emb_dim, feature_dim, l2norm=True)
        model.to(device)
        losses = train(true_dist, model, num_steps)
        print(num_starts, num_classes, feature_dim, "l2norm", losses[-1])

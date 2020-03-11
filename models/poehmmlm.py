import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import torch as th
import torch.nn as nn

import torch_struct as ts

from utils import Pack

from .misc import ResidualLayer, ResidualLayerOld
from .fflm import FfLm

class FfEmission(FfLm):
    def forward(self, input):
        input = self.prepare_input(input[:,:-1])
        # emb_x: batch x time x hidden
        emb_x = self.dropout(self.emb(input))
        B, T, H = emb_x.shape
        emb_x_T = emb_x.transpose(-1, -2)
        # cnn_out: batch x hidden x time -> batch x time x hidden
        cnn_out = self.cnn(emb_x_T).relu().transpose(-1, -2)
        logits = th.einsum(
            "nth,vh->ntv",
            self.mlp(self.dropout(cnn_out)),
            self.proj.weight,
        )
        # return batch x time x V
        # unnormalized for poe
        return logits

class PoeHmmLm(nn.Module):
    def __init__(self, V, config):
        super(PoeHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        if "old_res" in config and config.old_res: 
            # overwrite with old version           
            ResidualLayer = ResidualLayerOld       


        self.C = config.num_classes
        self.fb = foo.get_fb(self.C)

        self.semiring = getattr(ts, config.semiring)

        # p(z0)
        self.start_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )

        # p(zt | zt-1)
        self.state_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.trans_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, self.C),
        )

        # p(xt | zt)
        self.preterminal_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, len(V)),
        )
        self.ff = FfEmission(self.V, config)


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    @property
    def start(self):
        return self.start_mlp(self.start_emb).squeeze(-1).log_softmax(-1)
        #return self.start_mlp(self.start_emb).squeeze(-1).softmax(-1)

    @property
    def transition(self):
        return self.trans_mlp(self.state_emb).log_softmax(-1).permute(-1, -2) 
        #return self.trans_mlp(self.state_emb).softmax(-1).permute(-1, -2)

    @property
    def emission(self):
        return self.terminal_mlp(self.preterminal_emb).permute(-1, -2)

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        pass

    def ts_score(self, text, mask=None, lengths=None):
        N, T = text.shape
        emission = (self.ff(text).unsqueeze(-1) + self.emission).log_softmax(-2)
        log_potentials = ts.LinearChain.arhmm(
            transition = self.transition,
            emission = emission,
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        elbo = ts.LinearChain(ts.MaxSemiring).sum(log_potentials, lengths=lengths).sum()
        return Pack(
            evidence = elbo,
            elbo = elbo,
            loss = elbo,
        )

    def score(self, text, mask=None, lengths=None):
        if self.training:
            return self.ts_score(text, mask, lengths)
        N, T = text.shape
        emission = (self.ff(text).unsqueeze(-1) + self.emission).log_softmax(-2)
        log_potentials = ts.LinearChain.arhmm(
            transition = self.transition,
            emission = emission,
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        marginals, alphas, betas = self.fb(log_potentials.detach())
        evidence = alphas.gather(
            0,
            (lengths-1).view(1, N, 1).expand(1, N, self.C),
        ).logsumexp(-1).sum()
        elbo = (marginals.detach() * log_potentials)[mask[:,1:]].sum()
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )

import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm2.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


import torch as th
import torch.nn as nn

import torch_struct as ts

from utils import Pack

from .misc import ResidualLayer
from .fflm import FfLm

class FfEmission(FfLm):
    def __init__(self, V, config):
        # We cannot tie weights here since we would like
        # |Z| polytopes as in MoS (Yang 2017)
        assert not config.tie_weights
        super(FfEmission, self).__init__(V, config)

        self.Z = config.num_classes

        # over-write proj
        self.proj = nn.Linear(
            in_features = config.hidden_dim,
            out_features = len(V) * config.num_classes,
        )

    def forward(self, input, states):
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
        ).view(B, T-1, -1, self.Z)
        # return batch x time x V x Z
        return logits.log_softmax(-2)

    def forward_meh(self, input, states):
        input = self.prepare_input(input[:,:-1])
        # emb_x: batch x time x hidden
        emb_x = self.dropout(self.emb(input))
        B, T, H = emb_x.shape
        Z = states.shape[0]
        # broadcast emb_x: batch x time x hidden x z
        emb_x = emb_x.view(B, 1, T, H) + states.view(1, Z, 1, H)
        # emb_x_T: batch * z x hidden x time
        emb_x_T = emb_x.permute(0, 1, 3, 2).view(-1, H, T)
        cnn_out = self.cnn(emb_x_T).relu().transpose(-1, -2)
        logits = th.einsum(
            "nth,vh->ntv",
            self.mlp(self.dropout(cnn_out)),
            self.emb.weight if self.tie_weights else self.proj.weight,
        ).view(B, Z, T-1, -1)
        # return batch x time x V x Z
        return logits.log_softmax(-1).permute(0, 2, 3, 1)


class ArHmmLm(nn.Module):
    def __init__(self, V, config):
        super(ArHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.semiring = getattr(ts, config.semiring)
        self.fb = foo.get_fb(self.C)

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
        self.terminal_mlp = FfEmission(self.V, config)


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

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        pass

    def score_ts(self, text, mask=None, lengths=None):
        # p(X = x)
        emission = self.terminal_mlp(text, self.preterminal_emb)
        log_potentials = ts.LinearChain.arhmm(
            transition = self.transition,
            emission = emission,
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        # Perform tensor contraction online (instead of in memory)
        evidence = ts.LinearChain(self.semiring).sum(log_potentials, lengths=lengths)
        return evidence.sum()

    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        emission = self.terminal_mlp(text, self.preterminal_emb)
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

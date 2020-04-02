
import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import torch as th
import torch.nn as nn

import torch_struct as ts

#from .misc import ResidualLayer, ResidualLayerOld
from .misc import ResidualLayerOld, LogDropout

from utils import Pack

class HmmLm(nn.Module):
    def __init__(self, V, config):
        super(HmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        ResidualLayer = ResidualLayerOld

        self.C = config.num_classes

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
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, len(V)),
        )

        self.transition_dropout = LogDropout(config.transition_dropout)

        self.keep_counts = config.keep_counts > 0
        if self.keep_counts:
            self.register_buffer(
                "counts",
                th.zeros(self.C, len(self.V)),
            )
            self.register_buffer(
                "state_counts",
                th.zeros(self.C, dtype=th.int),
            )


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    @property
    def start(self):
        return self.start_mlp(self.start_emb).squeeze(-1).log_softmax(-1)

    @property
    def transition(self):
        return self.transition_dropout(
            self.trans_mlp(self.state_emb),
            column_dropout = True,
        ).log_softmax(-1).permute(-1, -2) 

    @property
    def emission(self):
        return self.terminal_mlp(self.preterminal_emb).log_softmax(-1).permute(-1, -2)

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        pass

    def score_ts(self, text, mask=None, lengths=None):
        # p(X = x)
        log_potentials = ts.LinearChain.hmm(
            transition = self.transition,
            emission = self.emission,
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        # Perform tensor contraction online (instead of in memory)
        evidence = ts.LinearChain(self.semiring).sum(log_potentials, lengths=lengths)
        #evidence = ts.LinearChain().sum(log_potentials, lengths=lengths)
        return evidence.sum()

    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        log_potentials = ts.LinearChain.hmm(
            transition = self.transition,
            emission = self.emission,
            init = self.start,
            observations = text,
            semiring = ts.LogSemiring,
        )
        marginals, alphas, betas, log_m = self.fb(log_potentials)
        evidence = alphas.gather(
            0,
            (lengths-1).view(1, N, 1).expand(1, N, self.C),
        ).logsumexp(-1).sum()
        elbo = (marginals.detach() * log_potentials)[mask[:,1:]].sum()

        if self.keep_counts and self.training:
            with th.no_grad():
                unary_marginals = th.cat([
                    log_m[:,0,None].logsumexp(-2),
                    log_m.logsumexp(-1),
                ], 1).exp()
                self.counts.index_add_(
                    1,
                    text.view(-1),
                    unary_marginals.view(-1, self.states_per_word).t(),
                )
                max_states = self.word2state[text[mask]].gather(
                    -1,
                    unary_marginals[mask].max(-1).indices[:,None],
                ).squeeze(-1)
                self.state_counts.index_add_(0, max_states, th.ones_like(max_states, dtype=th.int))

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )


import os
import time as timep

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "/home/justinchiu/code/python/genbmm/opt/hmm3.py"
    if os.getenv("LOCAL") is not None
    else "/home/jtc257/python/genbmm/opt/hmm3.py"
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import numpy as np

import torch as th
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropout

from utils import Pack
from assign import read_lm_clusters, assign_states_brown_cluster

import wandb

class DshmmLm(nn.Module):
    def __init__(self, V, config):
        super(DshmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.timing = config.timing > 0
        self.chp_theta = config.chp_theta > 0

        self.C = config.num_classes

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word

        self.num_layers = config.num_layers

        ResidualLayer = ResidualLayerOld

        """
        word2state, state2word = assign_states(
            self.C, self.states_per_word, len(self.V), self.words_per_state)
        """
        #num_clusters = 128 if config.assignment == "brown" else 64
        num_clusters = config.num_clusters if "num_clusters" in config else 128
        word2cluster, word_counts, cluster2word = read_lm_clusters(
            V,
            path=f"clusters/lm-{num_clusters}/paths",
        )
        self.word_counts = word_counts

        assert self.states_per_word * num_clusters <= self.C

        word2state = None
        if config.assignment == "brown":
            (
                word2state,
                cluster2state,
                word2cluster,
                c2sw_d,
            ) = assign_states_brown_cluster(
                self.C,
                word2cluster,
                V,
                self.states_per_word,
            )
        else:
            raise ValueError(f"No such assignment {config.assignment}")

        # need to save this with model
        self.register_buffer("word2state", th.from_numpy(word2state))
        self.register_buffer("cluster2state", th.from_numpy(cluster2state))
        self.register_buffer("word2cluster", th.from_numpy(word2cluster))

        self.tvm_fb = "tvm_fb" in config and config.tvm_fb
        #if self.states_per_word in [64, 128, 256, 512, 1024]:
        self.fb = foo.get_fb(self.states_per_word)

        # p(z0)
        self.start_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # p(zt | zt-1)
        self.state_emb = nn.Embedding(
            self.C, config.hidden_dim,
        )
        self.trans_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
        )
        #self.next_state_emb = nn.Embedding(self.C, config.hidden_dim)
        self.next_state_proj = nn.Linear(config.hidden_dim, self.C)

        # p(xt | zt)
        self.preterminal_emb = nn.Embedding(
            self.C, config.hidden_dim,
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, len(V)),
        )

        self.dropout = nn.Dropout(config.dropout)

        # tie embeddings key. use I separated pairs to specify
        # s: start
        # l: left
        # r: right
        # p: preterminal
        # o: output, can't be tied
        if "sl" in config.tw:
            self.state_emb.weight = self.start_emb
        if "lr" in config.tw:
            self.trans_mlp[-1].weight = self.state_emb.weight
        if "rp" in config.tw:
            self.preterminal_emb.weight = self.trans_mlp[-1].weight

        self.transition_dropout = LogDropout(config.transition_dropout)
        self.column_dropout = config.column_dropout > 0
        self.register_buffer(
            "drop_probs",
            th.empty(self.C).fill_(config.transition_dropout),
        )

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)

        self.keep_counts = config.keep_counts > 0
        if self.keep_counts:
            self.register_buffer(
                "counts",
                th.zeros(self.states_per_word, len(self.V)),
            )
            self.register_buffer(
                "state_counts",
                th.zeros(self.C, dtype=th.int),
            )

        self.register_buffer("zero", th.zeros(1))
        self.register_buffer("one", th.ones(1))


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    # don't permute here, permute before passing into torch struct stuff
    def start(self, states=None):
        start_emb = (self.start_emb[states]
            if states is not None
            else self.start_emb
        )
        return self.start_mlp(self.dropout(start_emb)).squeeze(-1).log_softmax(-1)

    def start_chp(self, states=None):
        start_emb = (self.start_emb[states]
            if states is not None
            else self.start_emb
        )
        return checkpoint(
            lambda x: self.start_mlp(self.dropout(x)).squeeze(-1).log_softmax(-1),
            start_emb
        )

    def transition_logits(self, states=None):
        state_emb = (self.state_emb.weight[states]
            if states is not None
            else self.state_emb.weight
        )
        next_state_proj = (self.next_state_proj.weight[states]
            if states is not None
            else self.next_state_proj.weight
        )
        x = self.trans_mlp(self.dropout(state_emb))
        return x @ next_state_proj.t()

    def transition_chp(self, states=None):
        state_emb = (self.state_emb.weight[states]
            if states is not None
            else self.state_emb.weight
        )
        next_state_proj = (self.next_state_proj.weight[states]
            if states is not None
            else self.next_state_proj.weight
        )
        return checkpoint(
            lambda x, y: (self.trans_mlp(self.dropout(x)) @ y.t()).log_softmax(-1),
            state_emb, next_state_proj,
        )


    def mask_transition(self, logits):
        # only in the weird case previously?
        # although now we may have unassigned states, oh well
        logits[:,-1] = float("-inf")
        logits = logits.log_softmax(-1)
        logits = logits.masked_fill(logits != logits, float("-inf"))
        logits[-1,:] = float("-inf")
        return logits

    def emission_logits(self, states=None):
        preterminal_emb = (self.preterminal_emb.weight[states]
            if states is not None
            else self.preterminal_emb.weight
        )
        logits = self.terminal_mlp(self.dropout(preterminal_emb))
        return logits

    def mask_emission(self, logits, word2state):
        a = self.a
        v = self.v

        i = th.stack([word2state.view(-1), a])
        C = logits.shape[0]
        sparse = th.sparse.ByteTensor(i, v, th.Size([C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
        log_probs = logits.masked_fill(~mask, -1e10).log_softmax(-1)
        log_probs = log_probs.masked_fill(log_probs != log_probs, float("-inf"))
        log_probs[-1,:] = float("-inf")
        return log_probs

    def emission_chp(self, word2state, states=None):
        preterminal_emb = (self.preterminal_emb.weight[states]
            if states is not None
            else self.preterminal_emb.weight
        )
        return checkpoint(
            lambda x: self.mask_emission(
                self.terminal_mlp(self.dropout(x)),
                word2state,
            ),
            preterminal_emb
        )

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        emission_logits = self.emission_logits
        word2state = self.word2state
        transition = self.mask_transition(self.transition_logits)
        emission = self.mask_emission(emission_logits, word2state)
        clamped_states = word2state[text]

        import pdb; pdb.set_trace()
        lpx = None
        return lpx

    def log_potentials(self, text, states=None):
        #word2state = self.word2state
        if self.timing:
            start_sample = timep.time()
        if states is not None:
            n = states.sum().item()
            s2d = states.cumsum(0) - 1
            s2d_pad = s2d.masked_fill(~states, n-1)
            word2state = s2d_pad[self.word2state]
        else:
            word2state = self.word2state

        if self.timing:
            print(f"total sample time: {timep.time() - start_sample}")
            start_compute = timep.time()

        if self.chp_theta:
            # only transition matrix?
            transition = self.transition_chp(states)
            #start = self.start_chp(states)
            #emission = self.emission_chp(word2state, states)
        else:
            transition_logits = self.transition_logits(states)
            transition = self.mask_transition(transition_logits)
        emission_logits = self.emission_logits(states)
        emission = self.mask_emission(emission_logits, word2state)
        start = self.start(states)

        #print(th.cuda.max_memory_allocated() / 2 ** 30)
        #print(th.cuda.max_memory_cached() / 2 ** 30)
        #print(text.nelement())

        if self.timing:
            print(f"total mask time: {timep.time() - start_mask}")
            start_clamp = timep.time()
        clamped_states = word2state[text]
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
            # oops a lot of padding
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        init = start[clamped_states[:,0]]
        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        if self.timing:
            print(f"total clamp time: {timep.time() - start_clamp}")
        return log_potentials.transpose(-1, -2)


    def score(self, text, mask=None, lengths=None, log_potentials=None):
        N, T = text.shape
        if self.timing:
            start_pot = timep.time()
        # sample states if training
        if self.training:
            states = ~self.drop_probs.bernoulli().bool()
            # padding state
            states[-1] = 1
        else:
            states = None
        log_potentials = (self.log_potentials(text, states)
            if log_potentials is None
            else log_potentials
        )
        if self.timing:
            print(f"total pot time: {timep.time() - start_pot}")
            start_marg = timep.time()
        fb = self.fb
        #fb = self.fb_train
        #marginals, alphas, betas, log_m = fb(log_potentials, mask=mask)
        with th.no_grad():
            log_m, alphas = fb(log_potentials.detach(), mask=mask)
        if self.timing:
            print(f"total marg time: {timep.time() - start_marg}")
        #import pdb; pdb.set_trace()
        evidence = alphas[lengths-1, th.arange(N)].logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()
        #import pdb; pdb.set_trace()
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
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )


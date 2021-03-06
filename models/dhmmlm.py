import time as timep

import os
import importlib.util
#spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
#spec = importlib.util.spec_from_file_location("get_fb", "/home/jtc257/python/genbmm/opt/hmm3.py")
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

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropoutM

from utils import Pack
from assign import read_lm_clusters, assign_states_brown, assign_states, assign_states_uneven_brown

import wandb

class DhmmLm(nn.Module):
    def __init__(self, V, config):
        super(DhmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.timing = config.timing > 0

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
            word2state = assign_states_brown(
                self.C,
                word2cluster,
                V,
                self.states_per_word,
            )
        elif config.assignment == "unevenbrown":
            word2state = assign_states_uneven_brown(
                self.C,
                word2cluster,
                V,
                self.states_per_word,
                word_counts,
                config.num_common,
                config.num_common_states,
                config.states_per_common,
            )
        elif config.assignment == "uniform":
            word2state = assign_states(
                self.C, self.states_per_word, len(self.V))
        elif config.assignment == "word2vec":
            word2cluster_np = np.load("clusters/kmeans-vecs/word2state-k128-6b-100d.npy")
            word2cluster = {i: x for i, x in enumerate(word2cluster_np[:,0])}

            word2state = assign_states_brown(
                self.C,
                word2cluster,
                V,
                self.states_per_word,
            )
        else:
            raise ValueError(f"No such assignment {config.assignment}")

        # need to save this with model
        self.register_buffer("word2state", th.from_numpy(word2state))

        self.tvm_fb = "tvm_fb" in config and config.tvm_fb
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
            nn.Linear(config.hidden_dim, self.C),
        )

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

        self.transition_dropout = config.transition_dropout
        # argument is unused
        self.log_dropout = LogDropoutM(config.transition_dropout)
        self.dropout_type = config.dropout_type

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


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    # don't permute here, permute before passing into torch struct stuff
    @property
    def start_logits(self):
        return self.start_mlp(self.dropout(self.start_emb)).squeeze(-1)

    def mask_start(self, x, mask=None):
        return self.log_dropout(x, mask).log_softmax(-1)

    @property
    def transition_logits(self):
        return self.trans_mlp(self.dropout(self.state_emb.weight))

    def mask_transition(self, logits, mask=None):
        # only in the weird case previously?
        # although now we may have unassigned states, oh well
        #logits[:,-1] = float("-inf")
        logits = self.log_dropout(logits, mask).log_softmax(-1)
        logits = logits.masked_fill(logits != logits, float("-inf"))
        return logits

    @property
    def emission_logits(self):
        logits = self.terminal_mlp(self.dropout(self.preterminal_emb.weight))
        return logits

    def mask_emission(self, logits, word2state):
        if word2state is None:
            return logits.log_softmax(-1)
        i = th.stack([word2state.view(-1), self.a])
        sparse = th.sparse.ByteTensor(i, self.v, th.Size([self.C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        log_probs = logits.masked_fill(~mask, -1e10).log_softmax(-1)
        #log_probs[log_probs != log_probs] = float("-inf")
        log_probs = log_probs.masked_fill(log_probs != log_probs, float("-inf"))
        return log_probs

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        emission_logits = self.emission_logits
        word2state = self.word2state
        transition = self.mask_transition(self.transition_logits)
        emission = self.mask_emission(emission_logits, word2state)
        clamped_states = word2state[text]

        lpx = None
        return lpx

    def log_potentials(self, text,
        lpz=None, last_states=None,
        start_mask=None, transition_mask=None,
    ):
        batch, time = text.shape
        clamped_states = self.word2state[text]

        if self.timing:
            start_compute = timep.time()
        emission_logits = self.emission_logits
        word2state = self.word2state
        transition_logits = self.transition_logits
        if self.timing:
            print(f"total compute time: {timep.time() - start_compute}")
            start_mask = timep.time()
        emission = self.mask_emission(emission_logits, word2state)
        transition = self.mask_transition(transition_logits, transition_mask)
        if self.timing:
            print(f"total mask time: {timep.time() - start_mask}")
            start_clamp = timep.time()
        if lpz is not None and last_states is not None:
            tmp = lpz[:,:,None] + transition[last_states]
            tmp = tmp.masked_fill(tmp < -1e10, -1e10)
            # this happens when full rows are -inf?
            start = tmp.logsumexp(1)
            #start = (lpz[:,:,None] + transition[last_states]).logsumexp(1)
            b_idx = th.arange(batch, device=self.device)                  
            init = start[b_idx[:,None], clamped_states[:,0]]              
        else:
            start = self.mask_start(self.start_logits, start_mask)
            init = start[clamped_states[:,0]]

        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        #import pdb; pdb.set_trace()
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        if self.timing:
            print(f"total clamp time: {timep.time() - start_clamp}")
        return log_potentials.transpose(-1, -2)


    def score(self, text,
        lpz=None, last_states=None,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        if self.timing:
            start_pot = timep.time()
        start_mask, transition_mask = None, None
        if not self.training:
            # no dropout
            pass
        elif self.dropout_type == "transition":
            transition_mask = (th.empty(self.C, self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "starttransition":
            transition_mask = (th.empty(self.C, self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
            start_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "column":
            transition_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "startcolumn":
            transition_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
            start_mask = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
        elif self.dropout_type == "state":
            m = (th.empty(self.C, device=self.device)
                .fill_(self.transition_dropout)
                .bernoulli_()
                .bool()
            )
            start_mask, transition_mask = m, m
        elif self.dropout_type == "cluster":
            raise NotImplementedError("Will try later if necessary, didn't work in mshmm")
            pass
        else:
            raise ValueError(f"Unsupported dropout type {self.dropout_type}")
        log_potentials = self.log_potentials(text,
            lpz, last_states,
            start_mask,
            transition_mask,
        )
        if self.timing:
            print(f"total pot time: {timep.time() - start_pot}")
            start_marg = timep.time()
        with th.no_grad():
            log_m, alphas = self.fb(log_potentials.detach(), mask=mask)
        if self.timing:
            print(f"total marg time: {timep.time() - start_marg}")

        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        last_words = text[idx, lengths-1]
        end_states = self.word2state[last_words]

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
        #if self.timing:
            #import pdb; pdb.set_trace()
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), end_states


import time as timep

import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import numpy as np

import torch as th
import torch.nn as nn

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropout

from utils import Pack
from assign import read_lm_clusters, assign_states_brown, assign_states, assign_states_uneven_brown

import wandb

class ShmmLm(nn.Module):
    def __init__(self, V, config):
        super(ShmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

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
            word2state, state2word = assign_states(
                self.C, self.states_per_word, len(self.V), self.words_per_state)
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

        self.transition_dropout = LogDropout(config.transition_dropout)
        self.column_dropout = config.column_dropout > 0
        self.start_dropout = LogDropout(config.start_dropout)

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
    def start(self):
        x = self.start_mlp(self.dropout(self.start_emb)).squeeze(-1)
        return self.start_dropout(x, column_dropout=True).log_softmax(-1)

    @property
    def transition_logits(self):
        return self.transition_dropout(
            self.trans_mlp(self.dropout(self.state_emb.weight)),
            column_dropout = self.column_dropout,
        )

    def mask_transition(self, logits):
        # only in the weird case previously?
        # although now we may have unassigned states, oh well
        logits[:,-1] = float("-inf")
        return logits.log_softmax(-1)

    @property
    def emission_logits(self):
        logits = self.terminal_mlp(self.dropout(self.preterminal_emb.weight))
        return logits

    def mask_emission(self, logits, word2state):
        i = th.stack([word2state.view(-1), self.a])
        sparse = th.sparse.ByteTensor(i, self.v, th.Size([self.C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
        log_probs = logits.masked_fill(~mask, float("-inf")).log_softmax(-1)
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

        import pdb; pdb.set_trace()
        lpx = None
        return lpx

    def log_potentials(self, text):
        #if wandb.run.mode == "dryrun":
            #start_emit = timep.time()
        emission_logits = self.emission_logits
        word2state = self.word2state
        #if wandb.run.mode == "dryrun":
            #print(f"total emit time: {timep.time() - start_emit}")
            #start_transm = timep.time()
        transition = self.mask_transition(self.transition_logits)
        #if wandb.run.mode == "dryrun":
            #print(f"total trans time: {timep.time() - start_transm}")
            #start_emitm = timep.time()
        emission = self.mask_emission(emission_logits, word2state)
        #if wandb.run.mode == "dryrun":
            #print(f"total emitm time: {timep.time() - start_emitm}")
            #start_clamp = timep.time()
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
        #import pdb; pdb.set_trace()
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        init = self.start[clamped_states[:,0]]
        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        #if wandb.run.mode == "dryrun":
            #print(f"total clamp time: {timep.time() - start_clamp}")
        return log_potentials.transpose(-1, -2)


    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        #if wandb.run.mode == "dryrun":
            #start_pot = timep.time()
        log_potentials = self.log_potentials(text)
        #if wandb.run.mode == "dryrun":
            #print(f"total pot time: {timep.time() - start_pot}")
            #start_marg = timep.time()
        marginals, alphas, betas, log_m = self.fb(log_potentials, mask=mask)
        #if wandb.run.mode == "dryrun":
            #print(f"total marg time: {timep.time() - start_marg}")
        evidence = alphas[lengths-1, th.arange(N)].logsumexp(-1).sum()
        elbo = (marginals.detach() * log_potentials)[mask[:,1:]].sum()
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


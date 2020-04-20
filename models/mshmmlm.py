
import time as timep

import importlib.util
#spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
spec = importlib.util.spec_from_file_location("get_fb", "/home/jtc257/python/genbmm/opt/hmm3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import numpy as np

import torch as th
import torch.nn as nn

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropout
from .misc import Index1, Index2

from utils import Pack
from assign import read_lm_clusters, assign_states_brown_cluster

import wandb
from pytorch_memlab import profile, MemReporter

def make_f(t):
    def f(x):
        from pytorch_memlab import MemReporter
        print(t)
        print(checkmem())
        import pdb; pdb.set_trace()
    return f

def checkmem():
    return(
        f"{th.cuda.memory_allocated() / 2**30:.2f}, {th.cuda.memory_cached() / 2 ** 30:.2f}, {th.cuda.max_memory_cached() / 2 ** 30:.2f}"
    )

class MshmmLm(nn.Module):
    def __init__(self, V, config):
        super(MshmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word

        self.num_layers = config.num_layers

        ResidualLayer = ResidualLayerOld

        self.timing = config.timing > 0

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
        self.register_buffer("c2sw_d", c2sw_d)
        self.register_buffer("word2state_d", self.c2sw_d[self.word2cluster])

        self.tvm_fb = "tvm_fb" in config and config.tvm_fb
        #if self.states_per_word in [64, 128, 256, 512, 1024]:
        self.fb_train = foo.get_fb(self.states_per_word // 2)
        self.fb_test = foo.get_fb(self.states_per_word)

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

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)

        self.states_per_word_d = self.states_per_word // 2

        self.ad = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word_d)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.vd = th.ones((len(self.V)) * self.states_per_word_d).to(self.device)

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

        self.word_dropout = config.word_dropout
        if self.word_dropout > 0:
            with th.no_grad():
                self.uniform_emission = self.get_uniform_emission(
                    self.word2state.to(self.device),
                )

    def get_uniform_emission(self, word2state):
        a = self.a
        v = self.v

        i = th.stack([word2state.view(-1), a])
        sparse = th.sparse.FloatTensor(i, v, th.Size([self.C, len(self.V)]))
        return sparse.to_dense().log().log_softmax(-1)

    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    # don't permute here, permute before passing into torch struct stuff
    #@profile
    def start(self, states=None):
        start_emb = (self.start_emb[states]
            if states is not None
            else self.start_emb
        )
        return self.start_mlp(self.dropout(start_emb)).squeeze(-1).log_softmax(-1)

    #@profile
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

    #@profile
    def mask_transition(self, logits):
        # only in the weird case previously?
        # although now we may have unassigned states, oh well
        #logits[:,-1] = float("-inf")
        return logits.log_softmax(-1)

    #@profile
    def emission_logits(self, states=None):
        preterminal_emb = (self.preterminal_emb.weight[states]
            if states is not None
            else self.preterminal_emb.weight
        )
        logits = self.terminal_mlp(self.dropout(preterminal_emb))
        return logits

    #@profile
    def mask_emission(self, logits, word2state):
        a = self.ad if self.training else self.a
        v = self.vd if self.training else self.v
        #a = self.ad
        #v = self.vd

        i = th.stack([word2state.view(-1), a])
        C = logits.shape[0]
        sparse = th.sparse.ByteTensor(i, v, th.Size([C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
        log_probs = logits.masked_fill_(~mask, float("-inf")).log_softmax(-1)
        #log_probs.register_hook(make_f("emission log probs"))
        #log_probs[log_probs != log_probs] = float("-inf")
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

    #@profile
    def clamp(
        self, text, start, transition, emission, word2state,
        uniform_emission = None, word_mask = None,
    ):
        clamped_states = word2state[text]
        #if wandb.run.mode == "dryrun":
            #import pdb; pdb.set_trace()
            # oops a lot of padding
        batch, time = text.shape
        timem1 = time - 1
        #print("trans index start")
        #print(checkmem())
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        """
        log_potentials = Index2.apply(
            transition,
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        )
        """
        #log_potentials.register_hook(make_f("trans log pots index"))
        #print(checkmem())
        #print("trans index end")
        #import pdb; pdb.set_trace()
        
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        init = start[clamped_states[:,0]]
        #init = Index1.apply(start, clamped_states[:,0])
        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        #obs = Index2.apply(emission, clamped_states[:,:,:,None], text[:,:,None,None])
        # word dropout == replace with uniform emission matrix (within cluster)?
        # precompute that and sample mask
        if uniform_emission is not None and word_mask is not None:
            unif_obs = uniform_emission[clamped_states[:,:,:,None], text[:,:,None,None]]
            obs[word_mask] = unif_obs[word_mask]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        #if wandb.run.mode == "dryrun":
            #print(f"total clamp time: {timep.time() - start_clamp}")
        #import pdb; pdb.set_trace()
        return log_potentials.transpose(-1, -2)

    #@profile
    def compute_parameters(self, word2state, states=None, word_mask=None):
        #print(f"compute params start {checkmem()}")
        start = self.start(states)
        #print(f"start {checkmem()}")
        #import pdb; pdb.set_trace()
        transition = self.mask_transition(self.transition_logits(states))
        #print(f"transition {checkmem()}")
        #import pdb; pdb.set_trace()
        emission = self.mask_emission(self.emission_logits(states), word2state)
        #print(f"emission {checkmem()}")
        #import pdb; pdb.set_trace()
        return start, transition, emission

    def log_potentials(self, text, states=None, word_mask=None):
        #word2state = self.word2state
        word2state = self.word2state_d if states is not None else self.word2state

        start, transition, emission = self.compute_parameters(word2state, states)
        #if wandb.run.mode == "dryrun":
            #print(f"total emitm time: {timep.time() - start_emitm}")
            #start_clamp = timep.time()
        if word_mask is not None:
            uniform_emission = (self.uniform_emission[states]
                if states is not None else self.uniform_emission)
        else:
            uniform_emission = None
        #print("Preclamp")
        #print(checkmem())
        #print("clamp")
        return self.clamp(
            text, start, transition, emission, word2state,
            uniform_emission, word_mask,
        )

    def compute_loss(
        self,
        log_potentials, mask, lengths,
        keep_counts = False,
    ):
        N = lengths.shape[0]
        fb = self.fb_train if self.training else self.fb_test
        #fb = self.fb_train
        log_m, alphas= fb(log_potentials, mask=mask)
        #if wandb.run.mode == "dryrun":
            #print(f"total marg time: {timep.time() - start_marg}")
        evidence = alphas[lengths-1, th.arange(N)].logsumexp(-1).sum()
        elbo = (log_m.exp() * log_potentials)[mask[:,1:]].sum()
        #import pdb; pdb.set_trace()
        if keep_counts:
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


    #@profile
    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        #if wandb.run.mode == "dryrun":
            #start_pot = timep.time()
        # sample states if training
        if self.training:
            I = (th.distributions.Gumbel(self.zero, self.one)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.states_per_word // 2, dim=-1)
                .indices
            )
            states = self.cluster2state.gather(1, I).view(-1)

            # word dropout. Kills (uniform) if mask == 1
            # TODO: factor this out into args (also need to factor out dropout prob lol)
            word_mask = th.empty(
                text.shape, dtype=th.float, device=self.device
            ).bernoulli_(0.1).bool() if self.word_dropout > 0 else None
        else:
            states = None
            word_mask = None

        #import pdb; pdb.set_trace()
        log_potentials = self.log_potentials(text, states, word_mask)
        #log_potentials.register_hook(make_f("log_potentials score"))
        #import pdb; pdb.set_trace()
        #if wandb.run.mode == "dryrun":
            #print(f"total pot time: {timep.time() - start_pot}")
            #start_marg = timep.time()
        fb = self.fb_train if self.training else self.fb_test
        with th.no_grad():
            log_m, alphas = fb(log_potentials.detach(), mask=mask)
            #log_m, alphas = fb(log_potentials, mask=mask)
        #if wandb.run.mode == "dryrun":
            #print(f"total marg time: {timep.time() - start_marg}")
        evidence = alphas[
            lengths-1, th.arange(N, device=self.device)
        ].logsumexp(-1).sum()
        # exp = 0.5G?
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()
        #elbo.register_hook(make_f("elbo"))
        #print("end of forward")
        #print(checkmem())
        #print(f"after fb {checkmem()}")
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
        #print(text.shape)
        #import sys; sys.exit()
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )

    def scoren(self, text, mask=None, lengths=None):
        N, T = text.shape
        #if wandb.run.mode == "dryrun":
            #start_pot = timep.time()
        # sample states if training
        if self.training:
            I = (th.distributions.Gumbel(self.zero, self.one)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.states_per_word // 2, dim=-1)
                .indices
            )
            states = self.cluster2state.gather(1, I).view(-1)
        else:
            states = None

        log_potentials = self.log_potentials(text, states)
        #if wandb.run.mode == "dryrun":
            #print(f"total pot time: {timep.time() - start_pot}")
            #start_marg = timep.time()
        fb = self.fb_train if self.training else self.fb_test
        #marginals, alphas, betas, log_m = fb(log_potentials, mask=mask)
        log_m, alphas = fb(log_potentials, mask=mask)
        evidence = alphas[lengths-1, th.arange(N)].logsumexp(-1)
        elbo = (log_m.exp() * log_potentials)[mask[:,1:]]
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


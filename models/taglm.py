
import time as timep
import os

from collections import Counter

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
from torch.utils.checkpoint import checkpoint

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropout
from .charcnn import CharLinear
#from .stateemb import StateEmbedding
from .stateemb import StateEmbedding2 as StateEmbedding

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

class FactoredHmmLm(nn.Module):
    """ Has both charcnn and factored state embs
    """
    def __init__(self, V, Vtag, config):
        super(FactoredHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.Vtag = Vtag
        self.device = config.device

        self.C = config.num_classes

        self.num_clusters = config.num_clusters

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word
        self.train_states_per_word = config.train_spw
        self.states_per_word_d = config.train_spw

        self.num_layers = config.num_layers

        ResidualLayer = ResidualLayerOld

        self.timing = config.timing > 0
        self.chp_theta = config.chp_theta > 0

        self.reset_eos = "reset_eos" in config and config.reset_eos > 0
        self.flat_clusters = "flat_clusters" in config and config.flat_clusters > 0

        """
        word2state, state2word = assign_states(
            self.C, self.states_per_word, len(self.V), self.words_per_state)
        """
        #num_clusters = 128 if config.assignment == "brown" else 64
        num_clusters = config.num_clusters if "num_clusters" in config else 128

        if "dataset" not in config:
            path = f"clusters/lm-{num_clusters}/paths"
        elif config.dataset == "ptb":
            lmstring = "lm" if not self.flat_clusters else "flm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wikitext2":
            #lmstring = "w2lm"
            lmstring = "w2flm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wikitext103":
            lmstring = "wlm"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        elif config.dataset == "wsj":
            lmstring = "sup-wsj"
            path = f"clusters/{lmstring}-{num_clusters}/paths"
        else:
            raise ValueError

        word2cluster, word_counts, cluster2word = read_lm_clusters(
            V,
            path=path,
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
                self.states_per_word_d,
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
        self.fb_train = foo.get_fb(self.train_states_per_word)
        self.fb_test = foo.get_fb(self.states_per_word)

        # p(z0)
        """
        self.start_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        """
        self.start_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
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
        """
        self.state_emb = nn.Embedding(
            self.C, config.hidden_dim,
        )
        """
        self.state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
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
        self.next_state_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )
        #self.next_state_proj = nn.Linear(config.hidden_dim, self.C)

        # p(xt | zt)
        """
        self.preterminal_emb = nn.Embedding(
            self.C, config.hidden_dim,
        )
        """
        self.preterminal_emb = StateEmbedding(
            self.C,
            config.hidden_dim,
            num_embeddings1 = config.num_clusters if config.state == "fac" else None,
            num_embeddings2 = config.states_per_word if config.state == "fac" else None,
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
            #nn.Linear(config.hidden_dim, len(V)),
        )
        self.terminal_proj = (
            nn.Linear(config.hidden_dim, len(V))
            if config.emit == "word"
            else CharLinear(config.char_dim, config.hidden_dim, V, config.emit_dims, config.num_highway)
        )

        self.tag_mlp = nn.Sequential(
            ResidualLayer(
                in_dim = 2*config.hidden_dim,
                out_dim = config.hidden_dim,
                dropout = config.dropout,
            ),
            nn.Dropout(config.dropout),
            #nn.Linear(config.hidden_dim, len(V)),
        )
        self.tag_proj = nn.Linear(config.hidden_dim, len(Vtag))

        self.dropout = nn.Dropout(config.dropout)

        # tie embeddings key. use I separated pairs to specify
        # s: start
        # l: left
        # r: right
        # p: preterminal
        # o: output, can't be tied
        if "sl" in config.tw:
            self.state_emb.share(self.start_emb)
        if "lr" in config.tw:
            self.next_state_emb.share(self.state_emb)
        if "rp" in config.tw:
            self.preterminal_emb.share(self.next_state_emb)

        self.transition_dropout = LogDropout(config.transition_dropout)
        self.column_dropout = config.column_dropout > 0

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)


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
        start_emb = self.start_emb(states)
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

    #@profile
    def transition_logits(self, states=None):
        state_emb = self.state_emb(states)
        next_state_emb = self.next_state_emb(states)
        x = self.trans_mlp(self.dropout(state_emb))
        return x @ next_state_emb.t()

    #@profile
    def mask_transition(self, logits):
        # only in the weird case previously?
        # although now we may have unassigned states, oh well
        #logits[:,-1] = float("-inf")
        return logits.log_softmax(-1)

    def transition_chp(self, states=None):
        raise NotImplementedError
        state_emb = (self.state_emb.weight[states]
            if states is not None
            else self.state_emb.weight
        )
        next_state_proj = (self.next_state_proj.weight[states]
            if states is not None
            else self.next_state_proj.weight
        )
        #import pdb; pdb.set_trace()
        return checkpoint(
            lambda x, y: (self.trans_mlp(self.dropout(x)) @ y.t()).log_softmax(-1),
            state_emb, next_state_proj,
        )

    #@profile
    def emission_logits(self, states=None):
        preterminal_emb = self.preterminal_emb(states)
        h = self.terminal_mlp(self.dropout(preterminal_emb))
        logits = self.terminal_proj(h)
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

    def tag_emission(self, text, states=None):
        N, T = text.shape
        # share preterminal emb with vocab.
        preterminal_emb = self.preterminal_emb(states)
        word_emb = (self.terminal_proj.weight[text]
            if self.config.emit == "word"
            else self.terminal_proj.get_embs()[text]
        )
        h = self.tag_mlp(self.dropout(
            th.cat([
                word_emb[:,:,None].expand(
                    N,T,preterminal_emb.shape[-2],word_emb.shape[-1],
                ),
                preterminal_emb,
            ], -1)
        ))
        return self.tag_proj(h).log_softmax(-1)

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
        self, text, tags,
        start, transition, emission, tag_emission,
        word2state,
        uniform_emission = None, word_mask = None,
        reset = None,
    ):
        clamped_states = word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        if reset is not None:
            eos_mask = text[:,:-1] == self.V["<eos>"]
            # reset words following eos
            reset_states = word2state[text[:,1:][eos_mask]]
            log_potentials[eos_mask] = reset[reset_states][:,None]
            #lp = log_potentials.clone()
        
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        b_idx = th.arange(batch, device=self.device)
        init = (
            start[clamped_states[:,0]]
            if start.ndim == 1
            else start[b_idx[:,None], clamped_states[:,0]]
        )

        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        if tags is not None:
            # tag_emission will be None
            tag_obs = self.tag_emission(text, clamped_states)
            tag_obs = tag_obs.gather(
                -1,
                tags[:,:,None,None].expand(obs.shape)
            )
            #tag_obs = tag_emission[clamped_states[:,:,:,None], tags[:,:,None,None]]
            obs += tag_obs
        # word dropout == replace with uniform emission matrix (within cluster)?
        # precompute that and sample mask
        if uniform_emission is not None and word_mask is not None:
            unif_obs = uniform_emission[
                clamped_states[:,:,:,None],
                text[:,:,None,None],
            ]
            obs[word_mask] = unif_obs[word_mask]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        #if wandb.run.mode == "dryrun":
            #print(f"total clamp time: {timep.time() - start_clamp}")
        #import pdb; pdb.set_trace()
        return log_potentials.transpose(-1, -2)

    def trans_to(self, from_states, to_states):
        state_emb = self.state_emb(from_states)
        next_state_proj = self.next_state_emb(to_states)
        x = self.trans_mlp(self.dropout(state_emb))
        return (x @ next_state_proj.t()).log_softmax(-1)

    #@profile
    def compute_parameters(self, word2state,
        states=None, word_mask=None,
        lpz=None, last_states=None,
    ):
        if self.chp_theta:
            transition = self.transition_chp(states)
            #emission = self.emission_chp(word2state, states)
            #start = self.start_chp(states)
            #return start, transition, emission
        else:
            transition = self.mask_transition(self.transition_logits(states))

        if last_states is None:
            start = self.start(states)
        else:
            # compute start from last_state
            start = (
                lpz[:,:,None] + self.trans_to(last_states, states)
            ).logsumexp(1)
            # hope this isn't too big
             
        emission = self.mask_emission(self.emission_logits(states), word2state)
        #tag_emission = self.tag_emission(states)
        # Do not compute here, since p(tag | word, state) is too expensive
        tag_emission = None
        return start, transition, emission, tag_emission

    def log_potentials(
        self, text, tags,
        states = None,
        lpz=None, last_states=None,
        word_mask=None,
    ):
        #word2state = self.word2state
        word2state = self.word2state_d if states is not None else self.word2state

        start, transition, emission, tag_emission = self.compute_parameters(
            word2state, states,
            word_mask,
            lpz, last_states,
        )
        # really should put this in compute_parameters
        reset = self.start(states) if self.reset_eos else None
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
        #
        return self.clamp(
            text, tags,
            start, transition, emission, tag_emission,
            word2state,
            uniform_emission, word_mask,
            reset = reset,
        )

    def compute_loss(
        self,
        log_potentials, mask, lengths,
        keep_counts = False,
    ):
        N = lengths.shape[0]
        fb = self.fb_train if self.training else self.fb_test
        log_m, alphas = fb(log_potentials, mask=mask)

        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1)

    #@profile
    def score(
        self, text, tags,
        lpz=None, last_states=None,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        if self.training:
            I = (th.distributions.Gumbel(self.zero, self.one)
                .sample(self.cluster2state.shape)
                .squeeze(-1)
                .topk(self.train_states_per_word, dim=-1)
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

        log_potentials = self.log_potentials(
            text,
            tags,
            states,
            lpz, last_states,
            word_mask,
        )
        fb = self.fb_train if self.training else self.fb_test
        with th.no_grad():
            log_m, alphas = fb(log_potentials.detach(), mask=mask)
        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        last_words = text[idx, lengths-1]
        c2s = states.view(self.config.num_clusters, -1)
        end_states = c2s[self.word2cluster[last_words]]

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), end_states


    def get_tags(
        self,
        text,
        start, transition, emission, tag_emission,
        word2state,
        mask=None, lengths=None,
    ):
        N, T = text.shape
        # None out tag information
        log_pots = self.clamp(
            text, None, start, transition, emission, None, word2state,
        )
        # edge marginals
        log_m, alphas = self.fb_test(log_pots, mask=mask)
        # log_m: N x T x zt x zt-1
        # unary marginals
        log_unary_marginals = th.cat([
            log_m[:,0,None].logsumexp(-2),
            log_m.logsumexp(-1),
        ], 1)
        clamped_states = word2state[text]
        emit = self.tag_emission(text, clamped_states)
        log_p_tag = (log_unary_marginals.unsqueeze(-1) + emit).logsumexp(-2)
        return log_p_tag

    def ffbs(
        self,
        text, tags,
        start, transition, emission, tag_emission,
        word2state,
        mask, lengths,
    ):
        N, T = text.shape
        # N x T x Zt x Zt-1
        log_pots = self.clamp(
            text, tags, start, transition, emission, tag_emission, word2state,
        )
        log_m, alpha = self.fb_test(log_pots, mask=mask)
        # log_m: N x T x Zt x Zt-1
        # alpha: T x N x Zt
        clamped_states = word2state[text]

        # alpha[0] = p(z0 | x0)
        alpha[0] = (
            start[clamped_states[:,0]]
            + emission[clamped_states[:,0], text[:,0,None]]
        )
        # alpha[t] = p(zt | x0:t)
        alpha = alpha.log_softmax(-1)

        # sample from p(zt | zt+1, x) propto p(zt | x)p(zt+1 | zt)
        # = p(zt | x1:t)p(zt+1 | zt)
        # = alpha[t]p(zt+1 | zt)
        # unbatched
        batch_states = []
        for n in range(N):
            T = lengths[n]-1
            states = []
            last_state = alpha[T,n].exp().multinomial(1)
            states.append(last_state)
            for t in range(T-1, -1, -1):
                last_state = (
                    alpha[t,n] + transition[clamped_states[n,t],last_state]
                ).softmax(-1).multinomial(1)
                states.append(clamped_states[n,t,last_state])
            # reverse and pad
            batch_states.append(
                list(reversed(states))
                + [self.Vtag.stoi["<pad>"]] * (lengths.max() - len(states))
            )
        return th.LongTensor(batch_states)


    def blocked_gibbs(
        self,
        text,
        start, transition, emission, tag_emission,
        word2state,
        mask=None, lengths=None,
        n_iters=100,
        take_every=5,
    ):
        # one sentence at a time for now
        N, T = text.shape
        #assert N == 1
        # initialize tags
        log_p_tag = self.get_tags(
            text,
            start, transition, emission, tag_emission,
            word2state,
            mask, lengths,
        )
        # start with max
        tags = log_p_tag.max(-1).indices
        # start counter
        counts = [Counter() for _ in range(N)]
        for i in range(n_iters):
            for t in range(T):
                # sample states use FFBS
                states = self.ffbs(
                    text, tags,
                    start, transition, emission, tag_emission,
                    word2state,
                    mask, lengths,
                )
                # sample tag conditioned on state
                tags[:,t] = tag_emission[states[:,t]].softmax(-1).multinomial(1).squeeze()
                for n in range(N):
                    counts[n][tuple(tags[n].tolist())] += 1
        return th.LongTensor([c.most_common(1)[0][0] for c in counts])


    def clamp_mt(
        self,
        t,
        text, tags,
        start, transition, emission, tag_emission,
        word2state,
        uniform_emission = None, word_mask = None,
        reset = None,
    ):
        clamped_states = word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        if reset is not None:
            eos_mask = text[:,:-1] == self.V["<eos>"]
            # reset words following eos
            reset_states = word2state[text[:,1:][eos_mask]]
            log_potentials[eos_mask] = reset[reset_states][:,None]
            #lp = log_potentials.clone()
        
        # this gets messed up if it's the same thing multiple times?
        # need to mask.
        b_idx = th.arange(batch, device=self.device)
        init = (
            start[clamped_states[:,0]]
            if start.ndim == 1
            else start[b_idx[:,None], clamped_states[:,0]]
        )

        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        if tags is not None:
            tag_obs_l = tag_emission[clamped_states[:,:t,:,None], tags[:,:t,None,None]]
            obs[:,:t] += tag_obs_l
            tag_obs_r = tag_emission[clamped_states[:,t+1:,:,None], tags[:,t+1:,None,None]]
            obs[:,t+1:] += tag_obs_r
        # word dropout == replace with uniform emission matrix (within cluster)?
        # precompute that and sample mask
        if uniform_emission is not None and word_mask is not None:
            unif_obs = uniform_emission[
                clamped_states[:,:,:,None],
                text[:,:,None,None],
            ]
            obs[word_mask] = unif_obs[word_mask]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        #if wandb.run.mode == "dryrun":
            #print(f"total clamp time: {timep.time() - start_clamp}")
        #import pdb; pdb.set_trace()
        return log_potentials.transpose(-1, -2)


    def collapsed_gibbs(
        self,
        text,
        start, transition, emission, tag_emission,
        word2state,
        mask, lengths,
        n_iters=100,
        take_every=5,
    ):
        # one sentence at a time for now
        N, T = text.shape
        #assert N == 1
        # initialize tags
        log_p_tag = self.get_tags(
            text,
            start, transition, emission, tag_emission,
            word2state,
            mask, lengths,
        )
        # start with max
        tags = log_p_tag.max(-1).indices
        # start counter
        counts = [Counter() for _ in range(N)]
        # add in initial
        for n in range(N):
            counts[n][tuple(tags[n].tolist())] += 1
        for i in range(n_iters):
            for t in range(T):
                # marginalize over p(z | x, y-t)
                # edge marginals
                log_potentials = self.clamp_mt(
                    t,
                    text, tags,
                    start, transition, emission, tag_emission,
                    word2state,
                )

                # get p(yt | y-t, x, z)
                tags_hat = self.get_tags(
                    text, start, transition, emission, tag_emission, word2state,
                    mask=mask, lengths=lengths,
                )
                # sample tag given all other tags
                tag = tags_hat[lengths > t,t].exp().multinomial(1).squeeze()
                tags[lengths > t,t] = tag

            for n in range(N):
                counts[n][tuple(tags[n].tolist())] += 1

        # rerank 10 most common
        K = 10
        log_probs = []
        k_tags = []
        for k in range(K):
            tags = th.LongTensor([
                c.most_common(K)[k if len(c) > k else 0][0]
                for c in counts
            ]).to(text.device)
            log_potentials = self.clamp(
                text, tags,
                start, transition, emission, tag_emission,
                word2state,
            )
            log_m, alphas = self.fb_test(log_potentials, mask=mask)
            idx = th.arange(N, device=self.device)
            alpha_T = alphas[lengths-1, idx]
            evidence = alpha_T.logsumexp(-1)

            log_probs.append(evidence)
            k_tags.append(tags)

        best_tags = th.stack(k_tags, 1)[idx,th.stack(log_probs, -1).max(-1).indices]
        return best_tags, counts

        #return th.LongTensor([c.most_common(1)[0][0] for c in counts]), counts





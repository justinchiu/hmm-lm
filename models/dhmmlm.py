
import time as timep

import importlib.util
spec = importlib.util.spec_from_file_location("get_fb", "/n/home13/jchiu/python/genbmm/opt/hmm3.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import torch as th
import torch.nn as nn

import torch_struct as ts

from .misc import ResidualLayerOld, ResidualLayerOpt, LogDropout

from utils import Pack
from assign import perturb_kmax

class DhmmLm(nn.Module):
    def __init__(self, V, config):
        super(DhmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word

        self.init_noise_scale = 1 if config.assignment_noise == "gumbel" else 0

        self.noise_dist = th.distributions.Gumbel(
            th.tensor([0], device=self.device, dtype=th.float32),
            th.tensor([self.init_noise_scale], device=self.device, dtype=th.float32),
        ) 

        self.num_layers = config.num_layers

        ResidualLayer = ResidualLayerOld

        if self.states_per_word in [64, 128, 256, 512, 1024]:
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
            #nn.Linear(config.hidden_dim, len(V)+1),
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

        """
        self.a = (th.arange(0, len(self.V)+1)[:, None]
            .expand(len(self.V)+1, self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)+1) * self.states_per_word).to(self.device)
        """
        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)


        self.keep_counts = config.keep_counts
        if self.keep_counts:
            self.register_buffer(
                "counts",
                #th.zeros(self.C, len(self.V)+1),
                th.zeros(self.C, len(self.V)),
            )
            self.posterior_weight = config.posterior_weight

    @property
    def noise_scale(self):
        return self.noise_dist.scale.item()

    @noise_scale.setter
    def noise_scale(self, scale):
        self.noise_dist.scale.fill_(scale)

    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    # don't permute here, permute before passing into torch struct stuff
    @property
    def start(self):
        return self.start_mlp(self.dropout(self.start_emb)).squeeze(-1).log_softmax(-1)

    @property
    def transition(self):
        #return self.trans_mlp(self.dropout(self.state_emb.weight)).log_softmax(-1)
        return self.transition_dropout(
            self.trans_mlp(self.dropout(self.state_emb.weight)),
            column_dropout = self.column_dropout,
        ).log_softmax(-1)

    def transition_sparse(self, states):
        return self.trans_mlp(self.dropout(self.state_emb(states))).log_softmax(-1)

    @property
    def emission_logits(self):
        logits = self.terminal_mlp(self.dropout(self.preterminal_emb.weight))
        #logits[:,-1] = float("-inf")
        return logits

    def mask_emission_old(self, logits, state2word):
        # manually mask each emission distribution
        mask = th.zeros_like(logits).scatter(
            -1,
            state2word,
            1,
        ).bool()

        self.a = (th.arange(0, len(self.V)+1)[:, None]
            .expand(len(self.V)+1, self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)+1) * self.states_per_word).to(self.device)

        #i = th.stack([self.a, self.word2state.view(-1)])
        i = th.stack([self.word2state.view(-1), self.a])
        #sparse = th.sparse.BoolTensor(i, v, torch.Size([self.C, len(self.V)+1]))
        sparse = th.sparse.ByteTensor(i, self.v, th.Size([self.C, len(self.V)+1]))
        mask2 = sparse.to_dense()

        return logits.masked_fill(~mask, float("-inf")).log_softmax(-1)

    def mask_emission(self, logits, word2state):
        #i = th.stack([self.word2state.view(-1), self.a])
        i = th.stack([word2state.view(-1), self.a])
        #sparse = th.sparse.ByteTensor(i, self.v, th.Size([self.C, len(self.V)+1]))
        sparse = th.sparse.ByteTensor(i, self.v, th.Size([self.C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        return logits.masked_fill(~mask, float("-inf")).log_softmax(-1)

    def emission_sparse(self, states):
        raise NotImplementedError
        unmasked_logits = self.terminal_mlp(self.dropout(self.preterminal_emb(states)))
        unmasked_logits[:,:,:,-1] = float("-inf")
        batch, time, k, v = unmasked_logits.shape
        logits = unmasked_logits.gather(-1, self.state2word[states])
        return logits.log_softmax(-1)

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        pass

    def log_potentials_sparse(self, text):
        raise NotImplementedError
        clamped_states = self.word2state[text]
        init = self.start[clamped_states[:,0]]
        # this is memory-bad
        transition = self.transition_sparse(clamped_states[:,:-1])
        batch, timem1, k, v = transition.shape
        log_potentials = transition.gather(
            -1,
            clamped_states[:,1:,None,:].expand(batch, timem1, k, k),
        )
        # this is memory-bad
        emission = self.emission_sparse(clamped_states)
        # this is memory-bad
        obs = emission[
            self.state2word[clamped_states] == text[:,:,None,None]
        ].view(batch, timem1+1, self.states_per_word, 1)

        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]

        return log_potentials.transpose(-1, -2)

    def assign_states(self, logits):
        return perturb_kmax(
            logits,
            self.noise_dist, # turn off during test...?
            self.states_per_word,
        )

    def log_potentials(self, text):
        #s = timep.time()
        emission_logits = self.emission_logits
        #word2state = self.assign_states(emission_logits)
        #word2state = self.assign_states(emission_logits.log_softmax(-1).log_softmax(0))
        word2state = self.assign_states(self.counts.log())
        #print(f"kmax: {timep.time()-s:.3f}")
        #s = timep.time()
        transition = self.transition
        #print(f"trans: {timep.time()-s:.3f}")
        #s = timep.time()
        emission = self.mask_emission(emission_logits, word2state)
        #print(f"emit mask: {timep.time()-s:.3f}")
        #s = timep.time()
        clamped_states = word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        init = self.start[clamped_states[:,0]]
        obs = emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        #print(f"clamp: {timep.time()-s:.3f}")
        return log_potentials.transpose(-1, -2), word2state


    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        #start_pot = timep.time()
        log_potentials, word2state = self.log_potentials(text)
        #log_potentials = self.log_potentials_time(text)
        #print(f"total pot time: {timep.time() - start_pot}")
        #start_fb = timep.time()
        marginals, alphas, betas, log_m = self.fb(log_potentials, mask=mask)
        #print(f"fb time: {timep.time() - start_fb}")
        #start_gather = timep.time()
        evidence = alphas.gather(
            0,
            (lengths-1).view(1, N, 1).expand(1, N, self.states_per_word),
        ).logsumexp(-1).sum()
        #print(f"evidence time: {timep.time() - start_gather}")
        elbo = (marginals.detach() * log_potentials)[mask[:,1:]].sum()
        if self.keep_counts and self.training:
            with th.no_grad():
                unary_marginals = th.cat([
                    log_m[:,0,None].logsumexp(-2),
                    log_m.logsumexp(-1),
                ], 1).exp()
                self.counts.view(-1).mul_(self.posterior_weight).index_add_(
                    0,
                    (word2state[text] + self.C * text[:,:,None]).view(-1),
                    unary_marginals.view(-1),
                )
                #print(self.counts.max())
                #import pdb; pdb.set_trace()
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )

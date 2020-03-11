
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
from assign import assign_states

class ChmmLm(nn.Module):
    def __init__(self, V, config):
        super(ChmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word

        self.num_layers = config.num_layers

        ResidualLayer = ResidualLayerOld

        word2state, state2word = assign_states(
        #word2state, state2word = assign_states3(
            self.C, self.states_per_word, len(self.V), self.words_per_state)

        # need to save this with model
        self.register_buffer("word2state", th.from_numpy(word2state))
        self.register_buffer("state2word", th.from_numpy(state2word))

        self.tvm_fb = "tvm_fb" in config and config.tvm_fb
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
            nn.Linear(config.hidden_dim, len(V)+1),
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

        self.keep_counts = config.keep_counts
        if self.keep_counts:
            self.register_buffer(
                "counts",
                th.zeros(self.states_per_word, len(self.V)+1),
            )


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
    def emission(self):
        unmasked_logits = self.terminal_mlp(self.dropout(self.preterminal_emb.weight))
        unmasked_logits[:,-1] = float("-inf")
        # manually mask each emission distribution
        mask = th.zeros_like(unmasked_logits).scatter(
            -1,
            self.state2word,
            1,
        ).bool()
        logits = unmasked_logits.masked_fill(~mask, float("-inf"))
        return logits.log_softmax(-1)

    def emission_sparse(self, states):
        unmasked_logits = self.terminal_mlp(self.dropout(self.preterminal_emb(states)))
        unmasked_logits[:,:,:,-1] = float("-inf")
        batch, time, k, v = unmasked_logits.shape
        logits = unmasked_logits.gather(-1, self.state2word[states])
        return logits.log_softmax(-1)

    def forward(self, inputs, state=None):
        # forall x, p(X = x)
        pass

    def log_potentials_memorybad(self, text):
        clamped_states = self.word2state[text]
        init = self.start[clamped_states[:,0]]
        # this is memory-bad
        transition = self.transition_sparse(clamped_states[:,:-1])
        import pdb; pdb.set_trace()
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

    def log_potentials_sparse(self, text):
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

    def log_potentials_time(self, text):
        start_transition = timep.time()
        transition = self.transition
        print(f"Transition time: {timep.time() - start_transition}s")
        clamped_states = self.word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        start_transition_index = timep.time()
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        print(f"Transition index time: {timep.time() - start_transition_index}s")
        start_init = timep.time()
        init = self.start[clamped_states[:,0]]
        print(f"Init time: {timep.time() - start_init}s")
        start_emit = timep.time()
        obs = self.emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        print(f"Emit time: {timep.time() - start_emit}s")
        start_add = timep.time()
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        print(f"Add time: {timep.time() - start_add}")
        return log_potentials.transpose(-1, -2)

    def log_potentials(self, text):
        transition = self.transition
        clamped_states = self.word2state[text]
        batch, time = text.shape
        timem1 = time - 1
        log_potentials = transition[
            clamped_states[:,:-1,:,None],
            clamped_states[:,1:,None,:],
        ]
        init = self.start[clamped_states[:,0]]
        obs = self.emission[clamped_states[:,:,:,None], text[:,:,None,None]]
        log_potentials[:,0] += init.unsqueeze(-1)
        log_potentials += obs[:,1:].transpose(-1, -2)
        log_potentials[:,0] += obs[:,0]
        return log_potentials.transpose(-1, -2)


    def score(self, text, mask=None, lengths=None):
        N, T = text.shape
        start_pot = timep.time()
        log_potentials = self.log_potentials(text)
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
                self.counts.index_add_(
                    1,
                    text.view(-1),
                    unary_marginals.view(-1, self.states_per_word).t(),
                )
        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        )

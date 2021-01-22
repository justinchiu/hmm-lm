import os

import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "hmm_runners/hmm.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

import torch as th
import torch.nn as nn

import torch_struct as ts

from .misc import ResLayer, LogDropoutM

from utils import Pack

from .linear_utils import get_2d_array, project_logits

class LHmmLm(nn.Module):
    def __init__(self, V, config):
        super(LHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device


        self.sm_emit = config.sm_emit
        self.sm_trans = config.sm_trans

        self.timing = config.timing > 0

        self.C = config.num_classes

        self.fb = foo.get_fb(self.C)
        self.word2state = None

        self.hidden_dim = config.hidden_dim

        # p(z0)
        self.start_emb = nn.Parameter(
            th.randn(config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, self.C),
        )

        # p(zt | zt-1)
        self.state_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.next_state_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )

        # p(xt | zt)
        self.preterminal_emb = nn.Parameter(
            th.randn(self.C, config.hidden_dim),
        )
        self.terminal_mlp = nn.Sequential(
            ResLayer(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
        )
        self.terminal_emb = nn.Parameter(
            th.randn(len(V), config.hidden_dim)
        )

        self.transition_dropout = config.transition_dropout
        self.log_dropout = LogDropoutM(config.transition_dropout)
        self.dropout_type = config.dropout_type

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

        # init
        for p in self.parameters():
            if p.dim() > 1:
                th.nn.init.xavier_uniform_(p)

        # log-linear or linear, etc
        self.parameterization = config.parameterization
        self.l2norm = config.l2norm
        self.anti = config.anti
        self.diffproj = config.diffproj
        if self.parameterization == "smp":
            if config.projection_method == "static":
                self._projection = nn.Parameter(self.init_proj())
                if not config.update_projection:
                    self._projection.requires_grad = False
                if self.diffproj:
                    self._projection_emit = nn.Parameter(self.init_proj())
                    if not config.update_projection:
                        self._projection_emit.requires_grad = False
            self.projection_method = config.projection_method


    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    def init_proj(self):
        if not self.anti:
            return get_2d_array(self.config.num_features, self.config.hidden_dim).t().to(self.device)
        else:
            projection_matrix = get_2d_array(
                self.config.num_features//2, self.config.hidden_dim).t().to(self.device)
            return th.cat([projection_matrix, -projection_matrix], -1)

    @property
    def projection(self):
        if self.projection_method == "static":
            pass
        elif self.projection_method == "random":
            self._projection = nn.Parameter(
                self.init_proj()
            )
            self._projection.requires_grad = False
        else:
            raise ValueError(f"Invalid projection_method: {self.projection_method}")
        return self._projection

    @property
    def projection_emit(self):
        if self.projection_method == "static":
            pass
        elif self.projection_method == "random":
            self._projection_emit = nn.Parameter(
                self.init_proj()
            )
            self._projection_emit.requires_grad = False
        else:
            raise ValueError(f"Invalid projection_method: {self.projection_method}")
        return self._projection_emit

    @property
    def start(self):
        return self.start_mlp(self.start_emb).log_softmax(-1)

    def start_logits(self):
        return self.start_mlp(self.start_emb).squeeze(-1)

    def mask_start(self, x, mask=None):
        return self.log_dropout(x, mask).log_softmax(-1)

    def transition(self, mask=None):
        fx = self.state_emb
        #gy = self.trans_mlp2(self.next_state_emb)
        #fx = self.state_emb
        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ self.next_state_emb.T
            #logits = fx @ gy.T
            logits = logits.log_softmax(-1)
            #logits = self.log_dropout(logits, mask).log_softmax(-1)
            #logits = logits.masked_fill(logits != logits, float("-inf"))
            return logits
        elif self.parameterization == "smp" and not self.sm_trans:
            # important to renormalize. maybe move this into project_logits
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = self.next_state_emb / self.next_state_emb.norm(dim=-1, keepdim=True)
            else:
                fy = self.next_state_emb
            logits = project_logits(
                fx[None],
                fy[None],
                self.projection,
                rff_method = self.config.rff_method,
            )[0].log_softmax(-1)
            #import pdb; pdb.set_trace()
            return logits
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

    def emission(self):
        fx = self.terminal_mlp(self.preterminal_emb)
        if self.parameterization == "softmax" or self.sm_emit:
            return (fx @ self.terminal_emb.T).log_softmax(-1)
        elif self.parameterization == "smp" and not self.sm_emit:
            # renormalize, important
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = self.terminal_emb / self.terminal_emb.norm(dim=-1, keepdim=True)
            else:
                fy = self.terminal_emb

            return project_logits(
                fx[None],
                fy[None],
                self.projection_emit if self.diffproj else self.projection,
            )[0].log_softmax(-1)
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")

    def forward(self, inputs, state=None):
        raise NotImplementedError
        # forall x, p(X = x)
        pass

    def score_ts(self, text, mask=None, lengths=None):
        # p(X = x)
        log_potentials = ts.LinearChain.hmm(
            transition = self.transition(),
            emission = self.emission(),
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        # Perform tensor contraction online (instead of in memory)
        evidence = ts.LinearChain(self.semiring).sum(log_potentials, lengths=lengths)
        #evidence = ts.LinearChain().sum(log_potentials, lengths=lengths)
        return evidence.sum()

    def log_potentials(self, text, states=None, lpz=None, last_states=None,):
        start_logits = self.start_logits()
        transition_logits = self.transition_logits()
        log_potentials = ts.LinearChain.hmm(
            #transition = self.mask_transition(
                #transition_logits,
                #None,
            #).t(),
            transition = self.transition().t(),
            emission = self.emission().t(),
            init = self.mask_start(
                start_logits,
                None,
            ),
            observations = text,
            semiring = ts.LogSemiring,
        )
        return log_potentials

    def compute_parameters(self,
        word2state=None,
        states=None, word_mask=None,       
        lpz=None, last_states=None,         
    ):
        #transition_logits = self.transition_logits()
        #transition = self.mask_transition(transition_logits, None)
        transition = self.transition()

        if lpz is not None:
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start_logits = self.start_logits()
            start = self.mask_start(start_logits, None)

        emission = self.emission()
        return start, transition, emission

    def clamp(                                              
        self, text, start, transition, emission, word2state=None,
        uniform_emission = None, word_mask = None,          
        reset = None,                                       
    ):
        return ts.LinearChain.hmm(
            transition = transition.t(),
            emission = emission.t(),
            init = start,
            observations = text,
        )

    def compute_loss(                                           
        self,                                                   
        log_potentials, mask, lengths,                          
        keep_counts = False,                                    
    ):                                                          
        N = lengths.shape[0]                                    
        #log_m, alphas = self.fb(log_potentials.clone(), mask=mask)
        log_m, alphas = self.fb(log_potentials.clone().float(), mask=mask)
                                                                
        idx = th.arange(N, device=self.device)            
        alpha_T = alphas[lengths-1, idx]                        
        evidence = alpha_T.logsumexp(-1).sum()                  
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()
                                                                
        return Pack(                                            
            elbo = elbo,                                        
            evidence = evidence,                                
            loss = elbo,                                        
        ), alpha_T.log_softmax(-1)                              


    def score(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        N, T = text.shape

        start_mask, transition_mask = None, None
        if not self.training or self.dropout_type == "none" or self.dropout_type is None:
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
        else:
            raise ValueError(f"Unsupported dropout type {self.dropout_type}")

        #transition_logits = self.transition_logits()
        #transition = self.mask_transition(transition_logits, transition_mask)
        transition = self.transition(transition_mask)
        emission = self.emission()

        if lpz is not None:
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start_logits = self.start_logits()
            start = self.mask_start(start_logits, start_mask)


        log_potentials = ts.LinearChain.hmm(
            transition = transition.t(),
            emission = emission.t(),
            init = start,
            observations = text,
            #semiring = ts.LogSemiring,
        )
        with th.no_grad():
            #log_m, alphas = self.fb(log_potentials.detach().clone(), mask=mask)
            log_m, alphas = self.fb(log_potentials.detach().clone().to(dtype=th.float32), mask=mask)
        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        if self.keep_counts and self.training:
            raise NotImplementedError("need to fix")
            with th.no_grad():
                unary_marginals = th.cat([
                    log_m[:,0,None].logsumexp(-2),
                    log_m.logsumexp(-1),
                ], 1).exp()
                self.counts.index_add_(
                    1,
                    text.view(-1),
                    unary_marginals.view(-1, self.C).t(),
                )
                max_states = unary_marginals[mask].max(-1).indices
                self.state_counts.index_add_(0, max_states, th.ones_like(max_states, dtype=th.int))

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), None

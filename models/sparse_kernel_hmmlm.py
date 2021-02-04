import os
import time as timep

"""
import importlib.util
spec = importlib.util.spec_from_file_location(
    "get_fb",
    "hmm_runners/hmm.py",
)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
"""

import torch as th
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

import torch_struct as ts

from genbmm import logbmm

from .misc import ResLayer, LogDropoutM

from utils import Pack

from .linear_utils import get_2d_array, project_logits

from assign import read_lm_clusters, assign_states_brown_cluster

from hmm_runners.hmm import get_fb
from hmm_runners.logmm2 import get_logmm_fwd, get_logmm_bwd
from hmm_runners.logmvv import get_logmvv_fwd, get_logmvv_bwd

def trans(s):
    return s.transpose(-2, -1).contiguous()

class SparseKernelHmmLm(nn.Module):
    def __init__(self, V, config):
        super(SparseKernelHmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device


        self.sm_emit = config.sm_emit
        self.sm_trans = config.sm_trans

        self.timing = config.timing > 0
        self.eff = config.eff

        self.C = config.num_classes
        self.D = config.num_features

        self.num_clusters = config.num_clusters

        self.words_per_state = config.words_per_state
        self.states_per_word = config.states_per_word
        self.train_states_per_word = config.train_spw
        self.states_per_word_d = config.train_spw

        self.fb = get_fb(self.C)
        self.fbd = get_fb(self.D)

        lmstring = "lm"
        num_clusters = config.num_clusters if "num_clusters" in config else 128
        path = f"clusters/{lmstring}-{num_clusters}/paths"
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

        # NO DROPOUT FOR NOW.

        # logmm setup
        if config.iterator == "bptt":
            raise NotImplementedError()
            self.N = config.bsz
            self.T = config.bptt
            logmm_ = get_logmm(self.N * self.T, self.C, self.D * self.D)
        elif config.iterator == "bucket":
            self.NT = config.bsz
            logmvv_fwd_ = get_logmvv_fwd(
                self.NT, self.states_per_word, self.D*self.D)
            logmvv_bwd_ = get_logmvv_bwd(
                self.NT, self.states_per_word, self.D*self.D)

            #logmvv_fwd_ = get_logmvv_fwd(self.D*self.D, self.C, self.NT)
            #logmvv_bwd_ = get_logmvv_bwd(self.D*self.D, self.C, self.NT)
        else:
            raise ValueError()

        # inner class
        class LogBmvv(th.autograd.Function):
            @staticmethod
            def forward(ctx, a, b, c):
                output = th.zeros(self.NT, self.D*self.D, 1, device=self.device)
                M = th.zeros(self.NT, self.D*self.D, 1, device=self.device)
                logmvv_fwd_(
                    a,
                    b,
                    c,
                    output,
                    M,
                )
                ctx.save_for_backward(a, b, c, output, M)
                import pdb; pdb.set_trace()
                return output

            @staticmethod
            def backward(ctx, grad_output):
                a, b, c, output, M = ctx.saved_tensors
                grad_a = th.empty(self.NT, self.C, device=self.device)
                grad_b = th.empty(self.NT, self.D, self.C, device=self.device)
                grad_c = th.empty(self.NT, self.D, self.C, device=self.device)
                logsumouterback_pytorch(a, b, c, output, M, grad_output, grad_a, grad_c)
                return grad_a, grad_b, grad_c


        def logmvv(a, b, c):
            N, T, C = a.shape
            _, _, _, D = b.shape
            # pad to constant N*T size, not important here but comes up with bucket batching.
            a_padded = th.cat([
                a.view(N*T, C),
                th.zeros(self.NT - N*T, C, device=self.device),
            ], 0)
            b_padded = th.cat([
                b.view(N*T, C, D),
                th.zeros(self.NT - N*T, C, D, device=self.device),
            ], 0)
            c_padded = th.cat([
                c.view(N*T, C, D),
                th.zeros(self.NT - N*T, C, D, device=self.device),
            ], 0)

            output = LogBmvv.apply(
                a_padded[:,None], # N * T x C
                b_padded.view(self.NT, C, D).transpose(-1, -2).contiguous(), # C x D
                c_padded.view(self.NT, C, D).transpose(-1, -2).contiguous(), # C x D
                # TODO: runs out of memory here
                # anything with dimension C needs to be chunked.
            )
            return output[:N*T,:,0].view(N, T, D, D)
        """
        def logmm(a, b):
            N, T, C = a.shape
            _, D, _ = b.shape
            # pad, can shove this somewhere else later
            a_padded = th.cat([
                a.view(-1, C),
                th.zeros(self.NT - N*T, C, device=self.device),
            ], 0)
            output = LogBmm.apply(
                a_padded[None], # N * T x C
                b.view(1, C, -1).transpose(-1, -2).contiguous(), # C x D * D
            )
            return output[0,:,:N*T].T.view(N, T, D, D)
        """

        self.logmvv = logmvv
        # /logmm setup


        self.hidden_dim = config.hidden_dim

        # p(z0)
        self.tie_start = config.tie_start
        self.start_emb = nn.Parameter(
            th.randn(config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
            ResLayer(config.hidden_dim, config.hidden_dim),
        )
        self.next_start_emb = nn.Parameter(
            th.randn(config.hidden_dim),
        )
        assert self.tie_start, "Needs tie_start to be correct"
        """
        if self.tie_start:
            # to prevent changing results, which previously had this bug
            # that was never seen since this parameter is not used
            # if start is tied.
            self.next_start_emb = nn.Parameter(
                th.randn(config.hidden_dim),
            )
        else:
            self.next_start_emb = nn.Parameter(
                th.randn(self.C, config.hidden_dim),
            )
        """

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

        self.a = (th.arange(0, len(self.V))[:, None]
            .expand(len(self.V), self.states_per_word)
            .contiguous()
            .view(-1)
            .to(self.device)
        )
        self.v = th.ones((len(self.V)) * self.states_per_word).to(self.device)



    def init_state(self, bsz):
        return self.start.unsqueeze(0).expand(bsz, self.C)

    def init_proj(self):
        #if self.config.rff_method == "relu":
            #return th.nn.init.xavier_uniform_(th.empty(self.config.hidden_dim, self.config.num_features)).to(self.device)
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

    def start(self, mask=None):
        #return self.start_mlp(self.start_emb).log_softmax(-1)
        fx = self.start_mlp(self.start_emb)
        fy = self.next_state_emb if self.tie_start else self.next_start_emb

        if self.parameterization == "softmax" or self.sm_trans:
            logits = fx @ fy.T
            logits = logits.log_softmax(-1)
            return logits
        elif self.parameterization == "smp" and not self.sm_trans:
            if self.l2norm:
                fx = fx / fx.norm(dim=-1, keepdim=True)
                fy = fy / fy.norm(dim=-1, keepdim=True)
            logits = project_logits(
                fx[None, None],
                fy[None],
                self.projection,
                rff_method = self.config.rff_method,
            )[0,0].log_softmax(-1)
            return logits
        else:
            raise ValueError(f"Invalid parameterization: {self.parameterization}")


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
        # Return sparsified emissions
        assert self.sm_emit, "only works with softmax emissions"
        word2state = self.word2state
        a = self.a
        v = self.v

        fx = self.terminal_mlp(self.preterminal_emb)
        logits = fx @ self.terminal_emb.T

        i = th.stack([word2state.view(-1), a])
        C = logits.shape[0]
        sparse = th.sparse.ByteTensor(i, v, th.Size([C, len(self.V)]))
        mask = sparse.to_dense().bool().to(logits.device)
        log_probs = logits.masked_fill_(~mask, float("-inf")).log_softmax(-1)
        return log_probs

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
        #start_logits = self.start_logits()
        #transition_logits = self.transition_logits()
        log_potentials = ts.LinearChain.hmm(
            #transition = self.mask_transition(
                #transition_logits,
                #None,
            #).t(),
            transition = self.transition().t(),
            emission = self.emission().t(),
            init = self.start(),
            #init = self.mask_start(
                #start_logits,
                #None,
            #),
            observations = text,
            semiring = ts.LogSemiring,
        )
        return log_potentials

    def compute_parameters(self,
        word2state=None,
        states=None, word_mask=None,       
        lpz=None, last_states=None,         
    ):
        # TODO: return struct instead of passing around distributions
        if self.eff:
            return self.compute_rff_parameters()

        transition = self.transition()

        if lpz is not None:
            start = (lpz[:,:,None] + transition[None]).logsumexp(1)
        else:
            start = self.start()

        emission = self.emission()
        return start, transition, emission

    def clamp(                                              
        self, text, start, transition, emission, word2state=None,
        uniform_emission = None, word_mask = None,
        reset = None,
        mask = None,
        lengths = None,
    ):
        # TODO: return struct instead of passing around distributions
        if self.eff:
            return self.clamp_rff(
                text, start, transition, emission,
                mask = mask,
                lengths = lengths,
            )

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
        if self.eff:
            # return two things, losses struct and next state vec
            return self.compute_rff_loss(log_potentials, mask, lengths)

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
            start = self.start(start_mask)
            #start_logits = self.start_logits()
            #start = self.mask_start(start_logits, start_mask)


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
            #log_m, alphas, betas = self.fb(log_potentials.detach().clone().to(dtype=th.float32), mask=mask)
        #import pdb; pdb.set_trace()
        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths-1, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        elbo = (log_m.exp_() * log_potentials)[mask[:,1:]].sum()

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), None

    def score_rff(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        N, T = text.shape
        C = self.C
        D = self.D

        start_mask, transition_mask = None, None
        # TODO: dropout?

        if self.timing:
            start_ = timep.time()
        emission = self.emission()
        if self.timing:
            print(f"total emit time: {timep.time() - start_}")
            start_ = timep.time()

        # gather emission
        # N x T x C
        logp_emit = emission[
            th.arange(self.C)[None,None],
            text[:,:,None],
        ]
        if self.timing:
            print(f"total emit index time: {timep.time() - start_}")
            start_ = timep.time()

        if self.l2norm:
            start_emb = self.start_emb / self.start_emb.norm(dim=-1, keepdim=True)
            state_emb = self.state_emb / self.state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = self.next_state_emb / self.next_state_emb.norm(dim=-1, keepdim=True)
        else:
            start_emb = self.start_emb
            state_emb = self.state_emb
            next_state_emb = self.next_state_emb

        # sum vectors and sum matrices
        log_phi_start = start_emb @ self.projection
        log_phi_w = state_emb @ self.projection
        log_phi_u = next_state_emb @ self.projection
        if self.timing:
            print(f"total phi projection emit time: {timep.time() - start_}")
            start_ = timep.time()

        # EFFICIENCY: anything involving C we want to checkpoint away
        ## First term
        if lpz is not None:
            log_start_vec = lpz
        else:
            # D
            log_sum_phi_u = log_phi_u.logsumexp(0)
            # SCALAR, logdot
            log_start_denominator = (log_phi_start + log_sum_phi_u).logsumexp(0)
            # D
            log_start_vec = log_phi_start - log_start_denominator
        if self.timing:
            print(f"total first term time: {timep.time() - start_}")
            start_ = timep.time()

        ## Middle terms
        # C = C x D + D
        log_denominator = (log_phi_w + log_sum_phi_u).logsumexp(-1)
        # C x Du x {Dw} + C x {Du} x Dw
        log_numerator = log_phi_u[:,:,None] + log_phi_w[:,None,:]

        log_trans_u = log_phi_u - log_denominator[:,None]
        log_trans_w = log_phi_w

        # C x Du x Dw
        log_trans_mat = log_numerator - log_denominator[:,None,None]
        if self.timing:
            print(f"total trans mat time: {timep.time() - start_}")
            start_ = timep.time()


        #if True:
        if not self.config.dbg_sparse:
            log_potentials = self.logmm(logp_emit, log_trans_mat)
        else:
            log_potentials = logbmm(
            #log_potentials_slow = logbmm(
                logp_emit.view(1, -1, C), # N * T x C
                log_trans_mat.view(1, C, -1), # C x D * D
            ).view(N, T, D, D)
            states = self.word2state[text]
            #print((log_potentials - log_potentials_slow).abs().max())
            logp_emit_sparse = emission[states, text[:,:,None]]
            log_trans_u_sparse = log_trans_u[states]
            log_trans_w_sparse = log_trans_w[states]
            log_potentials_sparse = self.logmvv(
                logp_emit_sparse,
                log_trans_u_sparse,
                log_trans_w_sparse,
            )
            import pdb; pdb.set_trace()
        
        if self.timing:
            print(f"total big logbmm time: {timep.time() - start_}")
            start_ = timep.time()
        log_eye = th.empty(D,D,device=self.device).fill_(float("-inf"))
        log_eye.diagonal().fill_(0.)
        log_potentials = th.where(mask[:,:,None,None], log_potentials, log_eye[None,None])
        if self.timing:
            print(f"total mask log pots time: {timep.time() - start_}")
            start_ = timep.time()

        log_end_vec = logbmm(
            logp_emit[None,th.arange(N),lengths-1],
            log_phi_u[None],
        )[0]
        if self.timing:
            print(f"total last term time: {timep.time() - start_}")
            start_ = timep.time()

        # approach 1
        log_potentials[th.arange(N), lengths-1,:,0] = log_end_vec
        log_potentials[th.arange(N), lengths-1,:,1:] = float("-inf")
        # approach 2 did not change results
        #log_potentials[th.arange(N), lengths-1] = log_eye
        #log_potentials[th.arange(N), lengths-1] += th.diag_embed(log_end_vec)
        log_potentials[:,0] += log_start_vec[None,:,None]
        # flip for torch_struct compat
        log_potentials = log_potentials.transpose(-1, -2)
        if self.timing:
            print(f"total mask time: {timep.time() - start_}")
            start_ = timep.time()

        with th.no_grad():
            #log_m, alphas = self.fb(log_potentials.detach().clone(), mask=mask)
            log_m, alphas = self.fbd(
            #log_m, alphas, betas = self.fbd(
                log_potentials.detach().clone().to(dtype=th.float32),
                #mask=th.cat([mask[:,(0,)],mask], dim=-1),
            )
        #import pdb; pdb.set_trace()
        if self.timing:
            print(f"total tvm time: {timep.time() - start_}")
            start_ = timep.time()
        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths, idx]
        #import pdb; pdb.set_trace()
        evidence = alpha_T.logsumexp(-1).sum()
        #import pdb; pdb.set_trace()
        # mask to get rid of nans
        log_potentials[log_potentials == float("-inf")] = 0
        #elbo = (log_m.exp() * log_potentials)[mask].sum()
        elbo = (log_m.exp_() * log_potentials)[mask].sum()
        #import pdb; pdb.set_trace()
        if self.timing:
            print(f"total loss time: {timep.time() - start_}")
            start_ = timep.time()

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1), None


    def compute_rff_parameters(self):
        if self.l2norm:
            start_emb = self.start_emb / self.start_emb.norm(dim=-1, keepdim=True)
            state_emb = self.state_emb / self.state_emb.norm(dim=-1, keepdim=True)
            next_state_emb = self.next_state_emb / self.next_state_emb.norm(dim=-1, keepdim=True)
        else:
            start_emb = self.start_emb
            state_emb = self.state_emb
            next_state_emb = self.next_state_emb

        # sum vectors and sum matrices
        log_phi_start = start_emb @ self.projection
        log_phi_w = state_emb @ self.projection
        log_phi_u = next_state_emb @ self.projection

        # start
        # D
        log_sum_phi_u = log_phi_u.logsumexp(0)
        # SCALAR, logdot
        log_start_denominator = (log_phi_start + log_sum_phi_u).logsumexp(0)
        # D
        log_start_vec = log_phi_start - log_start_denominator

        ## transition
        # C = C x D + D
        log_denominator = (log_phi_w + log_sum_phi_u).logsumexp(-1)
        # C x Du x {Dw} + C x {Du} x Dw
        log_numerator = log_phi_u[:,:,None] + log_phi_w[:,None,:]
        # C x Du x Dw
        log_trans_mat = log_numerator - log_denominator[:,None,None]

        emission = self.emission()

        return log_start_vec, (log_trans_mat, log_phi_u), emission

    def clamp_rff(
        self, text, start, transition, emission,
        mask=None, lengths=None,
    ):
        N, T = text.shape

        log_start_vec = start
        log_trans_mat, log_phi_u = transition

        logp_emit = emission[
            th.arange(self.C)[None,None],
            text[:,:,None],
        ]

        log_potentials = self.logmm(logp_emit, log_trans_mat)
        log_eye = th.empty(self.D,self.D,device=self.device).fill_(float("-inf"))
        log_eye.diagonal().fill_(0.)
        log_potentials = th.where(mask[:,:,None,None], log_potentials, log_eye[None,None])

        log_end_vec = logbmm(
            logp_emit[None,th.arange(N),lengths-1],
            log_phi_u[None],
        )[0]

        log_potentials[th.arange(N), lengths-1,:,0] = log_end_vec
        log_potentials[th.arange(N), lengths-1,:,1:] = float("-inf")
        log_potentials[:,0] += log_start_vec[None,:,None]
        # flip for torch_struct compatbility
        log_potentials = log_potentials.transpose(-1, -2)
        return log_potentials

    def compute_rff_loss(                                           
        self,                                                   
        log_potentials, mask, lengths,                          
    ):
        N = lengths.shape[0]                                    
        log_m, alphas = self.fbd(log_potentials.clone().float())

        idx = th.arange(N, device=self.device)
        alpha_T = alphas[lengths, idx]
        evidence = alpha_T.logsumexp(-1).sum()
        # mask to get rid of nans
        log_potentials[log_potentials == float("-inf")] = 0
        elbo = (log_m.exp_() * log_potentials)[mask].sum()

        return Pack(
            elbo = elbo,
            evidence = evidence,
            loss = elbo,
        ), alpha_T.log_softmax(-1)


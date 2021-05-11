
import torch as th
import torch_struct as ts

from hmm_runners.hmm import get_fb

class CountCollector:
    def __init__(self, model):
        self.start = model.start()
        self.transition = model.transition()
        self.emission = model.emission()
        self.device = model.device
        self.C = model.C
        self.log_counts = th.empty(self.C, device=self.device).fill_(float("-inf"))

        self.fb = get_fb(self.C)

    def collect_counts(self, text, mask=None, length=None):
        # debugging
        #text = text[:2, :8].contiguous()
        #mask = mask[:2, :8].contiguous()

        with th.no_grad():
            log_potentials = ts.LinearChain.hmm(
                transition = self.transition.t(),
                emission = self.emission.t(),
                init = self.start,
                observations = text,
            )

            log_m, alphas = self.fb(
                log_potentials.detach().clone().to(dtype=th.float32),
                mask=mask,
            )

            unary_log_marginals = th.cat([
                log_m[:,0,None].logsumexp(-2),
                log_m.logsumexp(-1),
            ], 1)

        with th.enable_grad():
            start = self.start
            transition = self.transition.exp()
            N, T = text.shape

            p_emit = self.emission[
                th.arange(self.C)[None,None],
                text[:,:,None],
            ]

            log_alphas = []
            alphas = []
            evidences = []
            alpha_un = start + p_emit[:,0] # {N} x C
            Ot = alpha_un.logsumexp(-1, keepdim=True)
            log_alpha = alpha_un - Ot
            alpha = log_alpha.exp()
            log_alphas.append(log_alpha)
            alphas.append(alpha)
            evidences.append(Ot)
            for t in range(T-1):
                # logbmm
                #alpha = (alpha[:,:,None] + transition[None] + p_emit
                alpha_un = (alpha @ transition).log() + p_emit[:,t+1]
                Ot = alpha_un.logsumexp(-1, keepdim=True)
                log_alpha = alpha_un - Ot
                alpha = log_alpha.exp()
                log_alphas.append(log_alpha)
                alphas.append(alpha)
                evidences.append(Ot)
            O = th.cat(evidences, -1)
            evidence = O[mask].sum(-1)

            betas = th.autograd.grad(evidence, log_alphas, allow_unused=True)
            import pdb; pdb.set_trace()

            meh = 1


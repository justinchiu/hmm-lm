import sys
import os
import time
import torch
import numpy as np

import tvm
from tvm import autotvm
import tvm.runtime
import tvm.te

sizes = [32, 64, 128, 256, 512, 1024, 2048]

@autotvm.template("zf_hmm_runner")
def zf_hmm_runner(b, t, z, f, dtype):

    #b = tvm.te.var("batch")
    #t = tvm.te.var("num_step")
    #z = tvm.te.var("state_size")
    #f = tv.te.var("num_features")


    # emission log probs
    X = tvm.te.placeholder((t, b, z), name="X", dtype=dtype)

    # projection from hidden to embedding
    Wzf = tvm.te.placeholder((z, f), name="Wzf", dtype=dtype)
    # projection from embedding to hidden
    Wfz = tvm.te.placeholder((f, z), name="Wfz", dtype=dtype)

    # alphas, forward state log probs
    s_z = tvm.te.placeholder((t, b, z))
    # gammas, forward embedded log values
    s_f = tvm.te.placeholder((t, b, f))
    s_state = (s_z, s_f)

    # start distribution
    s_init_z = tvm.te.placeholder((1, 1, z), name="s_init_z", dtype=dtype)
    s_init_f = tvm.te.compute((1, b, f), lambda a, b, c: 0.0)
    s_init = (s_init_z, s_init_f)

    # reduce axes
    # logmm z to f
    zzr = tvm.te.reduce_axis((0, z), name='zf_r')
    # max zf
    zzm = tvm.te.reduce_axis((0, f), name='zf_rm')
    # logmm f to z
    ffr = tvm.te.reduce_axis((0, z), name='fz_r')
    # max fz
    ffm = tvm.te.reduce_axis((0, f), name='fz_rm')

    # gamma
    Mg = tvm.te.compute(
        (t, b, f),
        lambda tt, bb, ff: tvm.te.max(
            s_z[tt-1, bb, zzm] + Wzf[zzm, ff],
            axis = zzm,
        ),
        name="Mg")
    Gm = tvm.te.compute(
        (t, b, f),
        lambda tt, bb, ff: tvm.te.sum(
            tvm.te.exp(s_z[tt-1, bb, zzr] + Wzf[zzr, ff]- Mg[t, bb, ff]),
            axis = zzr,
        ),
        name = "Gm",
    )
    G = tvm.te.compute(
        (t, b, f),
        lambda tt, bb, ff: tvm.te.log(Gm[tt, bb, ff]) + Mg[tt, bb, ff],
        name = "G",
    )

    # alpha
    Ma = tvm.te.compute(
        (t, b, z),
        lambda t, bb, zz: tvm.te.max(
            # wait im confused, this needs to fill in s_f
            s_f[tt, bb, ffm] + X[tt, bb, zz] + Wfz[zz, ff],
            axis=ffm,
        ),
        name="Ma")
    Am = tvm.te.compute(
        (t, b, f),
        lambda tt, bb, zz: tvm.te.sum(
            tvm.te.exp(s_f[tt, bb, ffr] + Wfz[zz, ffr] - Ma[tt, bb, zz]),
            axis = ffr,
        ),
        name = "Am",
    )
    A = tvm.te.compute(
        (t, b, z),
        lambda tt, bb, zz: tvm.te.log(Gm[tt, bb, zz]) + Mg[tt, bb, zz],
        name = "A",
    )

    # PRODUCE TUPLE (gamma_t, alpha_t) from (gamma_t-1, alpha_t-1)
    C = tvm.te.compute(
        (),
        lambda : ,
        name = "C",
    )

    # / conversion so far ends here

    s_scan = tvm.te.scan(s_init, C, s_state, inputs=[X])

    s = tvm.te.create_schedule(s_scan.op)
    #tvm.lower(s, [X], simple_mode=True )

    return s, [X, s_scan]

from tvm.contrib.dlpack import to_pytorch_func

# batch, time, num states, num features
def get_fb(b, t, z, f):
    """
    with autotvm.apply_history_best(f'best_hmm_k{size}.log'):
        with tvm.target.create("cuda"):
            s_mult, arg_bufs = hmm_runner('float32', size)
            mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
            hmm_pytorch = to_pytorch_func(mod)
    """
    with tvm.target.create("cuda"):
        s_mult, arg_bufs = zf_hmm_runner(b, t, z, f, 'float32')
        mod = tvm.build(s_mult, arg_bufs, target="cuda", target_host="llvm")
        hmm_pytorch = to_pytorch_func(mod)
    return hmm_pytorch

    # if the padding doesn't make a difference this must be an inclusive scan
    # x: batch x time x zt x zt-1
    #@profile
    def fb(x, Wzf, Wfz, mask=None):
        batch, time, size = x.shape

    return fb

if __name__ == "__main__":
    device = torch.device("cuda:0")
    V = 64
    H = 32
    N, T, Z, F = 3, 4, 16, 8
    text = torch.from_numpy(np.random.choice(V, size=(N, T))).to(device)

    state_emb = torch.randn(Z, H, device=device)
    next_state_emb = torch.randn(Z, H, device=device)
    projection = torch.randn(H, F, device=device)

    start = torch.randn(1, Z, device=device).log_softmax(-1)
    emission = torch.randn(Z, V, device=device).log_softmax(-1)

    # CxD
    log_phi_w = state_emb @ projection
    # CxD
    log_phi_u = next_state_emb @ projection
    # ^ Wfz, goes from hilbert space back to z

    # O(CD)
    log_denominator = (log_phi_w + log_phi_u.logsumexp(0, keepdim=True)).logsumexp(-1)
    # O(CD)
    normed_log_phi_w = log_phi_w - log_denominator[:,None]
    # ^ Wzf, goes from z to hilbert space

    # gather emission
    # N x T x C
    p_emit = emission[
        torch.arange(Z)[None,None],
        text[:,:,None],
    ]
    logmm = lambda x,y: (x[:,:,None] + y[None,:,:]).logsumexp(1)

    alphas = []
    gammas = []
    alpha = start + p_emit[:,0]
    alphas.append(alpha)
    for t in range(T-1):
        # gamma: N x D
        gamma = logmm(alpha, normed_log_phi_w)
        # alpha: N x C
        alpha = p_emit[:,t+1] + logmm(gamma, log_phi_u.T)

        alphas.append(alpha)
        gammas.append(gamma)
    evidence_manual = alpha.logsumexp(-1)
    betas = []
    xis = []
    # backward
    beta = torch.zeros(N, Z, device=device)#.fill_(math.log(1/C))
    betas.append(beta)
    for t in range(T-1,0,-1):
        # sanity check, beta_slow == beta
        #transition = (log_phi_w[:,None] + log_phi_u[None,:]).logsumexp(-1).log_softmax(-1)
        #beta_slow = logmm(p_emit[:,t] + beta, transition.T)

        xi = logmm(p_emit[:,t] + beta, log_phi_u)
        beta = logmm(xi, normed_log_phi_w.T)

        betas.append(beta)
        xis.append(xi)
    last_beta = beta + p_emit[:,0] + start
    # dont add last beta, not needed. can obtain from alpha[:,0] + beta[:,0]
    #betas.append(last_beta)
    aligned_betas = list(reversed(betas))
    aligned_xis = list(reversed(xis))
    # ground truth values, N x {T | T-1} x {Z | F}
    alpha = torch.stack(alphas, 1)
    gamma = torch.stack(gammas, 1)
    beta = torch.stack(aligned_betas, 1)
    xi = torch.stack(aligned_xis, 1)

    # TVM VERSION
    forward_tvm = get_fb(N, T, Z, F)
    # initialize buffers
    forward_tvm()


    """
    from tvm import autotvm
    import logging
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

    for size in sizes:
        task = autotvm.task.create(hmm_runner, args=('float32', size),
                                   target='cuda', target_host="llvm")

        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=5),
            runner=autotvm.LocalRunner(number=10, repeat=3, timeout=10, min_repeat_ms=50))


        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=100,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(f'hmm_k{size}.log')])

        autotvm.record.pick_best(f"hmm_k{size}.log", f"best_hmm_k{size}.log")
    """


import torch
import torch.nn as nn

import torch_struct as ts


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


class ResidualLayer(nn.Module):
    def __init__(
        self, in_dim = 100,
        out_dim = 100,
        dropout = 0.,
        # unused args
        do_norm = True,
        pre_norm = True,
        do_res = True,
    ):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = self.dropout(x)
        #x = self.dropout(self.lin1(x).relu())
        return self.layer_norm(self.dropout(self.lin2(x).relu()) + x)


class HmmLm(nn.Module):
    def __init__(self, V, config):
        super(HmmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device

        self.C = config.num_classes

        self.semiring = ts.LogSemiring

        # p(z0)
        self.start_emb = nn.Parameter(
            torch.randn(self.C, config.hidden_dim),
        )
        self.start_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )

        # p(zt | zt-1)
        self.state_emb = nn.Parameter(
            torch.randn(self.C, config.hidden_dim),
        )
        self.trans_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, self.C),
        )

        # p(xt | zt)
        self.preterminal_emb = nn.Parameter(
            torch.randn(self.C, config.hidden_dim),
        )
        self.terminal_mlp = nn.Sequential(
            ResidualLayer(config.hidden_dim, config.hidden_dim),
            nn.Linear(config.hidden_dim, len(V)),
        )

    @property
    def start(self):
        return self.start_mlp(self.start_emb).squeeze(-1).log_softmax(-1)

    @property
    def transition(self):
        return self.trans_mlp(self.state_emb).log_softmax(-1)

    @property
    def emission(self):
        return self.terminal_mlp(self.preterminal_emb).log_softmax(-1)

    def clamp(self, init, transition, emission, observations, semiring=ts.LogSemiring):
        V, C = emission.shape
        batch, N = observations.shape

        scores = semiring.one_(
            torch.empty(batch, N - 1, C, C, device=emission.device).type_as(emission)
        )
        scores[:, :, :, :] = semiring.times(scores, transition.view(1, 1, C, C))
        scores[:, 0, :, :] = semiring.times(
            scores[:, 0, :, :],
            init.view(1, 1, C) if init.ndim == 1 else init[:,None],
        )
        obs = emission[observations.view(batch * N), :]
        scores[:, :, :, :] = semiring.times(scores, obs.view(batch, N, C, 1)[:, 1:])
        scores[:, 0, :, :] = semiring.times(
            scores[:, 0], obs.view(batch, N, 1, C)[:, 0]
        )

        return scores

    def score(self, text, start=None, mask=None, lengths=None):
        log_potentials = self.clamp(
            transition = self.transition.t(),
            emission = self.emission.t(),
            init = self.start,
            observations = text,
            semiring = self.semiring,
        )
        evidence = ts.LinearChain(self.semiring).sum(log_potentials, lengths=lengths)
        return evidence.sum()

if __name__ == "__main__":
    V = 10
    device = torch.device("cuda:0")

    config = Pack(
        num_classes = 128,
        hidden_dim = 256,
        device = device,
    )
    model = HmmLm(
        [x for x in range(V)],
        config,
    )

    data = torch.randint(V, (16, 128), device=device)
    bptt = 16

    import time
    start = time.time()
    for x in data.split(bptt, dim=-1):
        model.zero_grad()
        n_tokens = x.nelement()
        loss = model.score(x.contiguous())
        loss.div(n_tokens).backward()
        # clip grad
        # optimizer stuff
    torch.cuda.synchronize()
    end = time.time()
    print(f"{end - start}s")

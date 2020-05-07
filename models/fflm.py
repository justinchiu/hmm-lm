
import torch as th
import torch.nn as nn

import torch_struct as ts

from utils import Pack

from .autoregressive import Autoregressive

class FfLm(ts.AutoregressiveModel):
    def __init__(self, V, config):
        super(FfLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device
        self.n = config.ngrams

        self.timing = False

        # default to weight tying
        self.tie_weights = config.tie_weights > 0 if "tie_weights" in config else True

        # padding for beginning of sentence
        self.register_buffer(
            "prefix",
            th.LongTensor([
                i for i in range(len(V), len(V) + config.ngrams - 1)
            ]),
        )
        self.prefix.requires_grad = False

        self.emb = nn.Embedding(
            num_embeddings = len(V) + config.ngrams - 1,
            embedding_dim = config.emb_dim,
            padding_idx = V["<pad>"],
        )

        self.cnn = nn.Conv1d(
            in_channels = config.emb_dim,
            out_channels = config.hidden_dim,
            kernel_size = config.ngrams-1,
        )
        self.dropout = nn.Dropout(config.dropout)
        m = []
        for i in range(config.num_layers-1):
            m.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            m.append(nn.ReLU())
            m.append(nn.Dropout(config.dropout))
        self.mlp = nn.Sequential(*m)
        if not self.tie_weights:
            self.proj = nn.Linear(
                config.hidden_dim,
                len(self.V),
                bias = False,
            )


    def forward(self, inputs):
        emb_x = self.dropout(self.emb(inputs))
        # emb_x_T: batch x hidden x time
        emb_x_T = emb_x.transpose(-1, -2)
        cnn_out = self.cnn(emb_x_T).relu().transpose(-1, -2)
        logits = th.einsum(
            "nth,vh->ntv",
            self.mlp(self.dropout(cnn_out)),
            self.emb.weight if self.tie_weights else self.proj.weight,
        )
        return logits.log_softmax(-1)

    def prepare_input(self, text, state=None):
        # append beginning tokens
        return th.cat([
            self.prefix.unsqueeze(0).expand(text.shape[0], self.n-1) if state is None else state,
            text,
        ], -1)


    # don't see the point of this right now, wrapping would be for beam search etc.
    def score(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        #state = self.init_state(text.shape[0])
        #input = text[:,:-1]
        input = self.prepare_input(text[:,:-1], last_states)
        #import pdb; pdb.set_trace()
        log_prob = self(input).gather(-1, text.unsqueeze(-1)).squeeze(-1)
        evidence = log_prob[mask].sum()
        return Pack(
            elbo = None,
            evidence = evidence.detach(),
            loss = evidence,
        ), None, text[:,-self.n+1:]

    def lpx(self, text, mask=None, lengths=None):
        input = self.prepare_input(text[:,:-1])
        return self(input).gather(-1, text.unsqueeze(-1)).squeeze(-1)


import torch as th
import torch.nn as nn

import torch_struct as ts

from .autoregressive import Autoregressive

from utils import Pack

class LstmLm(ts.AutoregressiveModel):
    def __init__(self, V, config):
        super(LstmLm, self).__init__()

        self.config = config
        self.V = V
        self.device = config.device
        self.timing = False

        self.emb = nn.Embedding(
            num_embeddings = len(V),
            embedding_dim = config.emb_dim,
            padding_idx = V["<pad>"],
        )
        self.lstm = nn.LSTM(
            input_size = config.emb_dim,
            hidden_size = config.hidden_dim,
            num_layers = config.num_layers,
            batch_first = True,
            dropout = config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

        # default to weight tying
        self.tie_weights = config.tie_weights > 0 if "tie_weights" in config else True

        if not self.tie_weights:
            self.proj = nn.Linear(
                config.hidden_dim,
                len(self.V),
                bias = False,
            )

        #self.init_params()

    # not helpful
    #def init_params(self):
        #for param in self.parameters():
            #param.data.uniform_(-0.1, 0.1)


    def init_state(self, bsz):
        return (
            th.zeros(bsz, self.lstm.num_layers, self.lstm.hidden_size, device=self.device),
            th.zeros(bsz, self.lstm.num_layers, self.lstm.hidden_size, device=self.device),
        )

    def convert_state(self, state):
        return (
            tuple(x.permute(1, 0, 2).contiguous() for x in state)
            if state is not None else state
        )

    def forward(self, inputs, state=None):
        emb_x = self.dropout(self.emb(inputs))
        #rnn_o, new_state = self.lstm(emb_x, self.convert_state(state))
        rnn_o, new_state = self.lstm(emb_x, state)
        logits = th.einsum(
            "nth,vh->ntv",
            self.dropout(rnn_o),
            self.emb.weight if self.tie_weights else self.proj.weight,
        )
        return logits.log_softmax(-1), new_state
        #return logits.log_softmax(-1), self.convert_state(new_state)

    # don't see the point of this right now, wrapping would be for beam search etc.
    def score_old(self, text, mask=None, lengths=None):
        state = self.init_state(text.shape[0])
        dist = Autoregressive(
            model = self,
            initial_state = state,
            n_classes = len(self.V),
            normalize = True,
            n_length = text.shape[1],
            start_class = self.V.stoi["<bos>"],
        )                     
        log_prob = dist.log_prob(
            text.unsqueeze(0), sparse=True, reduce=False, 
        ).squeeze(0).squeeze(-1)
        loss = log_prob[mask].sum()
        return Pack(
            evidence = loss,
            loss = loss,
            elbo = None,
        )

    def score(self, text, lpz=None, last_states=None, mask=None, lengths=None):
        # unpack `text` tuple
        input = text[:,:-1]
        output = text[:,1:]
        logits, state = self(input, last_states)
        log_px = logits.gather(-1, output[:,:,None]).squeeze(-1)
        loss = log_px[mask[:,1:]].sum()
        return Pack(
            evidence = loss,
            loss = loss,
            elbo = None,
        ), None, tuple(x.detach() for x in state)


    def lpx(self, text, mask=None, lengths=None):
        state = self.init_state(text.shape[0])
        dist = Autoregressive(
            model = self,
            initial_state = state,
            n_classes = len(self.V),
            normalize = True,
            n_length = text.shape[1],
            start_class = self.V.stoi["<bos>"],
        )                     
        return dist.log_prob(
            text.unsqueeze(0), sparse=True, reduce=False, 
        ).squeeze(0).squeeze(-1)

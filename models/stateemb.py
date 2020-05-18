import torch as th
import torch.nn as nn


class StateEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(StateEmbedding, self).__init__()
        self.num_embeddigns = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: torch.LongTensor(batch, time)

        pass

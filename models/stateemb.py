import torch as th
import torch.nn as nn

from .misc import ResidualLayerOld

class StateEmbedding(nn.Module):
    def __init__(self,
        num_embeddings,
        embedding_dim,
        num_embeddings1 = None,
        num_embeddings2 = None,
    ):
        super(StateEmbedding, self).__init__()
        self.num_embeddigns = num_embeddings
        self.embedding_dim = embedding_dim

        self.factored = num_embeddings1 is not None and num_embeddings2 is not None

        if not self.factored:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            assert num_embeddings == num_embeddings1 * num_embeddings2
            self.dim1 = num_embeddings2
            self.dim2 = num_embeddings2
            self.emb1 = nn.Embedding(num_embeddings1, embedding_dim)
            self.emb2 = nn.Embedding(num_embeddings2, embedding_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Tanh(),
            )

    def forward(self, x=None):
        # x: torch.LongTensor(batch, time) or None, the states
        if not self.factored:
            return self.emb(x) if x is not None else self.emb.weight
        if x is not None:
            # inner dim is emb2
            x1 = self.emb1(x // self.dim2)
            x2 = self.emb2(x % self.dim2)
            y = self.mlp(th.cat([x1, x2], -1))
            return y
        else:
            # construct cross product
            xprod = th.cat([
                self.emb1.weight[:,None].expand(self.dim1, self.dim2, self.embedding_dim),
                self.emb2.weight[None,:].expand(self.dim1, self.dim2, self.embedding_dim),
            ], -1)
            return self.mlp(xprod.view(self.dim1 * self.dim2, 2 * self.embedding_dim))


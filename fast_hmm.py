import torch
n_states = 127
n_words = 1269
n_steps = 20
alpha_init = torch.randn(n_states).log_softmax(-1)
emissions = torch.randn(n_states, n_steps).log_softmax(-1) # this should use gather from actual words
#words = torch.randint(low=0, high=n_words, size=(n_steps,)).view(-1)
# HMM
transition_matrix = torch.randn(n_states, n_states).log_softmax(-1)
alpha1 = alpha_init
for t in range(n_steps):
    alpha1 = (emissions[:,t] + transition_matrix.exp() @ alpha1.exp()).log_softmax(-1)

n_features = 67
pre_embeddings = torch.randn(n_states, n_features)
post_embeddings = torch.randn(n_states, n_features)
# Fast HMM
alpha2 = alpha_init
for t in range(n_steps):
    inner = (alpha2.view(-1, 1) + emissions[:,t].view(-1, 1) + post_embeddings).exp().sum(0) # , n_features
    alpha2 = (pre_embeddings @ inner).log_softmax(-1)

import pdb; pdb.set_trace()

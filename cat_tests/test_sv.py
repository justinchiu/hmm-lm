
# check rank of low rank decomposition with normalization
import torch

n = 256
d = 64

temp = 0.5

A = torch.randn(n, d)
B = torch.randn(n, d)
C = ((A @ B.T) / temp).softmax(-1)
u, s, v = C.svd()
print("softmax")
print((s > 1e-3).sum())

A = torch.randn(n, d)
B = torch.randn(n, d)
C = ((A.exp() @ B.T.exp()).log() / temp).softmax(-1)
u, s, v = C.svd()
print("wrong temp = polynomial kernel?")
print((s > 1e-3).sum())

A = torch.randn(n, d) / temp
B = torch.randn(n, d) / temp
C = ((A.exp() @ B.T.exp()).log()).softmax(-1)
u, s, v = C.svd()
print("normal temp")
print((s > 1e-3).sum())

import pdb; pdb.set_trace()

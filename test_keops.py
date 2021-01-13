
import torch
from pykeops.torch import LazyTensor

bsz = 1
D = 256
num_features = 1024

x = torch.randn(bsz, D, num_features)
y = torch.randn(bsz, D, num_features)

def logbmm(x, y):
    expand = x[:,:,None,:] + y[:,None,:,:]
    return expand.logsumexp(-1)

result = logbmm(x, y)

X = LazyTensor(x[0,:,None])
Y = LazyTensor(y[0,None])

Z = X + Y
output = (Z - Z.max(-1)).exp().sum(-1).log()

import pdb; pdb.set_trace()
# logsumexp is broken in keops?
output2 = (X + Y).logsumexp_reduction(-1)


"""
X = 1
Y = 1

# ... whats the point if have to expand
X = x.view(bsz, D, 1, num_features).expand(bsz, D, D, num_features).contiguous().view(-1, num_features)
Y = y.view(bsz, 1, D, num_features).expand(bsz, D, D, num_features).contiguous().view(-1, num_features)

print(X.shape)
print(Y.shape)

formula = "x + y"
variables = [
    f"x = Vi({num_features})",
    f"y = Vi({num_features})",
]

routine = generic_logsumexp("x + y", "a = Vi(1)", f"x = Vi({num_features})", f"y = Vi({num_features})")
output = routine(X, Y)
"""

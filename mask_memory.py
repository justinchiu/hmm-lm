
import torch

from pytorch_memlab import profile, MemReporter

def hmm_pytorch(a, b):
    pass

def log_eye(K, dtype, device):                                                
    x = torch.empty(K, K, dtype = dtype, device = device)                     
    x.fill_(float("-inf"))                                                    
    x.diagonal().fill_(0)                                                     
    return x                                                                  
                                                                              
def log_eye_cat(x):                                                           
    K = x.shape[-1]                                                           
    batch = x.shape[1]                                                        
    return torch.cat([                                                        
        x,                                                                    
        log_eye(K, x.dtype, x.device).view(1, 1, K, K).expand(1, batch, K, K),
    ], dim=0)                                                                 


batch = 16
time = 32
size = 512

device = torch.device("cuda:0")

@profile
def f():
    x = torch.randn(batch, time, size, size, device=device)
    mask = None

    x = x.permute(1, 0, 3, 2)

    lex = log_eye(size, dtype=x.dtype, device=x.device)
    out_fb = torch.empty(2, time+1, batch, size, device=x.device)
    out_fb.fill_(float("-inf"))                                  
    inp = torch.empty(time+1, batch, size, size, device=x.device)
    inp[-1] = lex                                                
    # forward                                                    
    inp[:-1] = x                                                 
    hmm_pytorch(inp, out_fb[0])                                  
    # backward                                                   
    inp[range(time-1, -1, -1)] = x.transpose(-2, -1)             
    hmm_pytorch(inp, out_fb[1])                                  

    alphas = out_fb[0]
    betas = out_fb[1].flip(0)

    log_marginals = x
    log_marginals += alphas[:-1].view(time, batch, size, 1)
    log_marginals += betas[1:].view(time, batch, 1, size)
    log_marginals -= alphas[-1].logsumexp(-1).view(1, -1, 1, 1)

    if mask is not None:
        #marginals.masked_fill_(~mask[1:,:,None,None], 0)
        log_marginals.masked_fill_(~mask[1:,:,None,None], float("-inf"))
    log_marginals = log_marginals.permute(1, 0, 3, 2)
    return log_marginals

x = f()
y = g()
reporter = MemReporter()
reporter.report()

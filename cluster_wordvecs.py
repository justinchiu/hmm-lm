
import sys

import math
import time

from collections import Counter

from pathlib import Path

import numpy as np

import torch as th

import torchtext
from datasets.ptb import PennTreebank, BucketIterator

import faiss

chp_path = "wandb_checkpoints/shmm_k16384_wps512_spw128_ed256_d256_dp0.0_tdp0.5_cdp1_tokens_b1024_adamw_lr0.01_c5.0_tw_nas0_pw1_asunevenbrown_nc1024_ncs8191_spc8/12044_4.90.pth"
chp_path2 = "wandb_checkpoints/ff_k16384_wps512_spw128_ed256_d256_dp0.3_tdp0_cdp0_tokens_b1024_adamw_lr0.001_c5.0_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/92647_5.08.pth"
chp_path3 = "wandb_checkpoints/lstm_k16384_wps512_spw128_ed256_d256_dp0.3_tdp0_cdp0_tokens_b512_adamw_lr0.001_c5_tw_nas0_pw1_asbrown_nc0_ncs0_spc0/93550_4.60.pth"

chp = th.load(chp_path)
# chp["args"] will have the args eventually...
#config = get_args()
config = chp["args"]

device = th.device("cuda:0")
config.device = device

TEXT = torchtext.data.Field(batch_first = True)
train, valid, test = PennTreebank.splits(
    TEXT,
    newline_eos = True,
)

# move to larger one later
TEXT.build_vocab(train, vectors="glove.6B.100d")
V = TEXT.vocab
wordvecs = V.vectors.numpy()

k = 128
d = 100

res = faiss.StandardGpuResources()
config = faiss.GpuIndexFlatConfig()
config.useFloat16 = False
config.device = 0
index = faiss.GpuIndexFlatL2(res, d, config)

kmeans = faiss.Clustering(d, k)
kmeans.verbose = True
kmeans.niter = 32
kmeans.train(wordvecs, index)

# simpler api
# kmeans = faiss.Kmeans(d, k, gpu=True)
# kmeans.train(states)
 
centroids_np = faiss.vector_to_array(kmeans.centroids).reshape(k, d)

# squared distances and indices
D, I = index.search(wordvecs, 1)

train_loss = faiss.vector_to_array(kmeans.obj)[-1]

print(f"K-means k{k} d{d}: train_loss {train_loss}")
np.save(
    "clusters/kmeans-vecs/word2state-k128-6b-100d.npy",
    I,
)
#import pdb; pdb.set_trace()

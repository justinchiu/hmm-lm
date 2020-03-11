

import numpy as np
import torch as th
import faiss

def convert_states(flat_states, idxs):
    state_idxs = [[] for _ in range(max(idxs) + 1)]
    for idx, state_idx in zip(idxs, flat_states.squeeze()):
        state_idxs[idx].append(state_idx)
    return state_idxs

states_path = "checkpoints/basic/states.pth"
states_dict = th.load(states_path)


states_np = [x.numpy() for x in states_dict["states"]]
flat_states_np = np.concatenate(states_np) 
words = states_dict["words"]
flat_idxs = [y for z in [
    [idx] * len(x)
    for idx, x in zip(states_dict["idxs"], states_dict["states"])
] for y in z]

valid_states_np = [x.numpy() for x in states_dict["valid_states"]]
valid_flat_states_np = np.concatenate(valid_states_np) 
valid_words = states_dict["valid_words"]
valid_flat_idxs = [y for z in [
    [idx] * len(x)
    for idx, x in zip(states_dict["valid_idxs"], states_dict["valid_states"])
] for y in z]

d = flat_states_np.shape[1]
#k = 2 ** 12
k = 2 ** 14

for k in [2 ** exponent for exponent in range(12, 18)]:
    print(f"Dimension: {d} | Num centroids: {k:,}")

    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16 = False
    config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, config)

    kmeans = faiss.Clustering(d, k)
    kmeans.verbose = True
    kmeans.niter = 32
    kmeans.train(flat_states_np, index)

    # simpler api
    # kmeans = faiss.Kmeans(d, k, gpu=True)
    # kmeans.train(states)
     
    centroids_np = faiss.vector_to_array(kmeans.centroids).reshape(k, d)

    # squared distances and indices
    D, I = index.search(flat_states_np, 1)
    D_valid, I_valid = index.search(valid_flat_states_np, 1)

    train_loss = faiss.vector_to_array(kmeans.obj)[-1]
    valid_loss = D_valid.sum()

    print(f"K-means k{k} d{d}: train_loss {train_loss} | valid_loss {valid_loss}")
    np.save(
        f"checkpoints/basic/centroids/centroids-k{k}-d{d}-t{train_loss:.2f}-v{valid_loss:.2f}",
        centroids_np,
    )
    states_np = convert_states(I, flat_idxs)
    np.save(
        f"checkpoints/basic/centroids/centroids-k{k}-d{d}-t{train_loss:.2f}-v{valid_loss:.2f}.states",
        states_np,
    )

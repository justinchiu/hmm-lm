import math

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

def nonnegative_softmax_kernel_feature_creator(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float=0.0001,
    log = False,
):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """

    #ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    ratio = 1.0

    bsz = data.size(0)
   
    projection = projection_matrix.unsqueeze(0).expand(bsz, -1, -1)

    # Compute wx
    # data:       bsz, len, D
    # projection: bsz, D, #features
    data_dash = torch.bmm(
        data,
        projection
    ) # bsz, len, #features

    # Compute ||x||^2/2
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, -1) # (bsz, len) ||x||^2
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(-1) # bsz, len, 1

    if log:
        if is_query:
            return (data_dash - diag_data
                - torch.max(data_dash, dim=-1, keepdim=True)[0]
                + math.log(eps)
            )
        else:
            return (data_dash - diag_data
                - torch.max(data_dash)
                + math.log(eps)
            )

    # Compute exp(wx - ||x||^2/2)  
    # (Lemma 1, SM(x, y) = E_{w~N(0,I)} exp(wx - ||x||^2/2) exp(wy - ||y||^2/2))
    if is_query:
        # for each query, we can independently scale to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True)[0]) + eps)
    else:
        # for keys, we need to use the same normalizer to avoid overflows
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

def get_2d_array(nb_rows, nb_columns, scaling=0):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn(nb_columns, nb_columns)
        q, _ = torch.qr(unstructured_block)
        q = q.T
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        unstructured_block = torch.randn(nb_columns, nb_columns)
        q, _ = torch.qr(unstructured_block)
        q = q.T
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.cat(block_list, 0)
    #print (final_matrix.size())

    if scaling == 0:
        multiplier = torch.norm(
            torch.randn(nb_rows, nb_columns), dim=-1).view(-1, 1)
    elif scaling == 1:
        multiplier = torch.sqrt(float(nb_columns)) * torch.ones((nb_rows))
    else:
        raise ValueError('Scaling must be one of {0, 1}. Was %s' % scaling)

    return multiplier * final_matrix

def keops_logbmm(x, y):
    # wow keops sucks.
    expand = x[:,:,None,:] + y[:,None,:,:]
    result = expand.logsumexp(-1)
    return result

def logbmm(x, y):
    expand = x[:,:,None,:] + y[:,None,:,:]
    return expand.logsumexp(-1)

def project_logits(query, key, projection_matrix, eps=0.0001, rff_method="log"):
    kernel = nonnegative_softmax_kernel_feature_creator

    if rff_method == "exp":
        query_features = kernel(query, projection_matrix, is_query=True, eps=eps)
        key_features = kernel(key, projection_matrix, is_query=False, eps=eps)
        values = query_features.bmm(key_features.transpose(-1, -2))
        return values.log()
        #exp_logits = values / values.sum(-1, keepdim=True)
        #return exp_logits.log()
    elif rff_method == "mixed":
        query_features = kernel(query, projection_matrix, is_query=True, eps=eps)
        key_features = kernel(key, projection_matrix, is_query=False, eps=eps)
        expand = (query_features[:,:,None] * key_features[:,None]).log()
        logits = expand.logsumexp(-1)

        # values.log == logits
        #values = query_features.bmm(key_features.transpose(-1, -2))
        #import pdb; pdb.set_trace()
        # both lead to 0 gradients
        
        return logits
    elif rff_method == "log":
        # log space
        log_query_features = kernel(query, projection_matrix, is_query=True, eps=eps, log=True)
        log_key_features = kernel(key, projection_matrix, is_query=False, eps=eps, log=True)
        # slow and memory...would like log-bmm
        # bxz x src x tgt x dim

        return checkpoint(logbmm, log_query_features, log_key_features)

        # chunking
        #N = 256
        #num_features = 
        #import pdb; pdb.set_trace()
        #for 
        """
        expand = log_query_features[:,:,None,:] + log_key_features[:,None,:,:]
        logits = (
            #log_query_features[:,:,None,:] + log_key_features[:,None,:,:]
            expand
        ).logsumexp(-1)
        #import pdb; pdb.set_trace()
        return logits
        """

    else:
        raise ValueError(f"Invalid rff_method: {rff_method}")


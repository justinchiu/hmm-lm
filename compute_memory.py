

words = 10000
#num_states = 2 ** 15
num_states = 2 ** 14
spw = 1024
spw = 512

def mem(x):
    return x * 4 / (2 ** 30)

def param_memory(words, num_states, hidden_dim):
    # all the activations in the MLPs
    start_mlp = num_states * hidden_dim * 6
    trans_mlp = num_states * hidden_dim * 6
    emit_mlp = num_states * hidden_dim * 6

    # all the matrices (2x because of logits then softmax)
    start = num_states * 2
    trans = num_states ** 2 * 2
    emit = num_states * words * 2
    #emit = num_states * words * 3

    return (
        start_mlp + start,
        trans_mlp + trans,
        emit_mlp + emit,
    )

def index_memory(num_words, spw):
    transition = num_words * spw 

def chain_memory(num_words, num_states):
    return (
        # log potentials and copy of the log potentials for reversing and giving to backward
        num_words * num_states ** 2 * 2
        # output of forward and backward (and backward.flip)
        + num_words * num_states * 3
        # log_marginals.exp() * log_pots (may not be necessary)
        + num_words * num_states ** 2 * 2
    )

print(num_states, spw)
print([mem(x) for x in param_memory(words, num_states, 512)])
print(mem(sum(param_memory(words, num_states, 512))))
print(mem(chain_memory(512, spw)))

## memory leaks in mask_emission and indexing into transition, ~2G

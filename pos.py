
from pathlib import Path

from collections import Counter

txtf = Path(".data/PTB/ptb.nopunct.txt")
tagf = Path(".data/PTB/ptb.nopunct.tags")

#clustersf = Path("viterbi_clusters/mshmm_k32768_nb128/paths")
#statesf = Path("viterbi_output/ptb_bucket_mshmm_k32768_wps512_spw256_tspw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtnone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc0.viterbi.txt")
#clustersf = Path("viterbi_clusters/brown_45/paths")
clustersf = Path("viterbi_clusters/brown_mik_45/paths")
statesf = txtf

# compare states and words
txt = txtf.read_text().strip().split("\n")
states = statesf.read_text().strip().split("\n")
tags = tagf.read_text().strip().split("\n")

# states has an additional EOS state
txt = [x.strip().split() for x in txt]
states = [x.strip().split()[:-1] for x in states]
#states = [x.strip().split() for x in states]
tags = [x.strip().split() for x in tags]

assert all([len(x) == len(y) for x,y in zip(txt, states)])
assert all([len(x) == len(y) for x,y in zip(txt, tags)])


# construct cluster mapping
state2cluster = {
    triple[1]: triple[0]
    # cluster, state, counts
    for triple in [
        line.strip().split("\t")
        for line in clustersf.read_text().strip().split("\n")
    ] 
}
clusters = [
    [
        state2cluster[state]
        for state in state_seq
    ] for state_seq in states
]

# compute cluster+tag co-occurrences
cluster_set = set(x for xs in clusters for x in xs)
cooc = {
    cluster: Counter() for cluster in cluster_set
}
for cts in zip(clusters, tags):
    for c, t in zip(*cts):
        cooc[c][t] += 1
# map cluster to most common co-occurring tag
cluster2tag = {
   cluster: tag_counts.most_common(1)[0][0]
   for cluster, tag_counts in cooc.items()
}

tags_hat = [
    [
        cluster2tag[cluster]
        for cluster in cluster_seq
    ] for cluster_seq in clusters
]

import pdb; pdb.set_trace()

# write out tags_hat
#outf = Path("viterbi_output/tags.out")
outf = Path("viterbi_output/tags.blah.out")
outf.write_text("\n".join([" ".join(xs) for xs in tags_hat]))

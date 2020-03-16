#!/bin/bash

input=.data/penn-treebank/ptb.train.txt

if [[ "$1" == "pmi" ]]; then
    cluster=/n/rush_lab/jc/code/tan-clustering/pmi_cluster.py
    python $cluster $input clusters/pmi/ptb.train.clusters > clusters/pmi/log
elif [[ "$1" == "lm256" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 256 --output_dir clusters/lm-256
elif [[ "$1" == "lm128" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 128 --output_dir clusters/lm-128
elif [[ "$1" == "lm64" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 64 --output_dir clusters/lm-64
else
    echo "Improper argument"
fi


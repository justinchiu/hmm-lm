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
elif [[ "$1" == "lm32" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 32 --output_dir clusters/lm-32
elif [[ "$1" == "lm16" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 16 --output_dir clusters/lm-16
elif [[ "$1" == "lm10" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 10 --output_dir clusters/lm-10
elif [[ "$1" == "lm8" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 8 --output_dir clusters/lm-8
elif [[ "$1" == "lm4" ]]; then
    l_cluster=/n/rush_lab/jc/code/brown-cluster/wcluster
    $l_cluster --text $input --c 4 --output_dir clusters/lm-4
else
    echo "Improper argument"
fi


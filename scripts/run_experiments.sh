#!/bin/bash

function run_lstms {
    source scripts/experiments.sh
    lstm_b4_d256 0 &
    lstm_b32_d256 1 &
    lstm_b4_d512 2 &
    lstm_b32_d512 3 &
}

function run_ffs {
    source scripts/experiments.sh
    ff_b4_d256_k5 0 &
    ff_b32_d256_k5 1 &
    ff_b4_d512_k5 2 &
    ff_b32_d512_k5 3 &
}

function run_ff_ks {
    source scripts/experiments.sh
    ff_b32_d256_k5 3 &
    ff_b32_d256_k4 0 &
    ff_b32_d256_k3 1 &
    ff_b32_d256_k2 2 &
}

function run_hmms {
    source scripts/experiments.sh
    hmm_b4_d256_k128 0 &
    hmm_b4_d512_k128 3 &
    hmm_b4_d256_k256 1 &
    hmm_b4_d512_k256 2 &
}

function run_hmms_overfit {
    source scripts/experiments.sh
    hmm_b4_d256_k128_overfit 0 &
    hmm_b4_d512_k128_overfit 1 &
    hmm_b4_d256_k256_overfit 2 &
    hmm_b4_d512_k256_overfit 3 &
}

function run_hmms_tvm {
    source scripts/experiments.sh
    hmm_b4_d256_k128_tvm 0 &
    hmm_b4_d256_k256_tvm 1 &
    hmm_b4_d256_k512_tvm 2 &
    hmm_b4_d256_k1024_tvm 3 &
}

function run_hmms_seed {
    source scripts/experiments.sh
    hmm_b4_d256_k512_s1234 0 &
    hmm_b4_d256_k512_s1 1 &
    hmm_b4_d256_k512_s2357 2 &
    hmm_b4_d256_k512_s2468 3 &
}

function run_hmms_512 {
    source scripts/experiments.sh
    hmm_b4_d512_k128_oldres 0 &
    hmm_b4_d512_k256_oldres 1 &
    hmm_b4_d512_k512_oldres 2 &
    hmm_b4_d512_k1024_oldres 3 &
}

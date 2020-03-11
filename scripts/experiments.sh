#!/bin/bash
# Lstm sweep
function lstm_b4_d256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d256.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b4_d256 \
        > logs/lstm_b4_d256.log
}
function lstm_b4_d512 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d512.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b4_d512 \
        > logs/lstm_b4_d512.log
}
function lstm_b4_d650 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d650.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b4_d650 \
        > logs/lstm_b4_d650.log
}
function lstm_b32_d256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d256.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b32_d256 \
        > logs/lstm_b32_d256.log
}
function lstm_b32_d512 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d512.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b32_d512 \
        > logs/lstm_b32_d512.log
}
function lstm_b32_d650 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/lstm-d650.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save lstm_b32_d650 \
        > logs/lstm_b32_d650.log
}
# Ff sweep
function ff_b4_d256_k2 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k2.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d256_k2 \
        > logs/ff_b4_d256_k2.log
}
function ff_b4_d256_k3 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k3.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d256_k3 \
        > logs/ff_b4_d256_k3.log
}
function ff_b4_d256_k4 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k4.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d256_k4 \
        > logs/ff_b4_d256_k4.log
}
function ff_b4_d256_k5 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k5.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d256_k5 \
        > logs/ff_b4_d256_k5.log
}
function ff_b4_d512_k2 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k2.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d512_k2 \
        > logs/ff_b4_d512_k2.log
}
function ff_b4_d512_k3 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k3.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d512_k3 \
        > logs/ff_b4_d512_k3.log
}
function ff_b4_d512_k4 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k4.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d512_k4 \
        > logs/ff_b4_d512_k4.log
}
function ff_b4_d512_k5 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k5.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b4_d512_k5 \
        > logs/ff_b4_d512_k5.log
}
function ff_b32_d256_k2 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k2.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d256_k2 \
        > logs/ff_b32_d256_k2.log
}
function ff_b32_d256_k3 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k3.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d256_k3 \
        > logs/ff_b32_d256_k3.log
}
function ff_b32_d256_k4 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k4.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d256_k4 \
        > logs/ff_b32_d256_k4.log
}
function ff_b32_d256_k5 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d256-k5.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d256_k5 \
        > logs/ff_b32_d256_k5.log
}
function ff_b32_d512_k2 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k2.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d512_k2 \
        > logs/ff_b32_d512_k2.log
}
function ff_b32_d512_k3 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k3.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d512_k3 \
        > logs/ff_b32_d512_k3.log
}
function ff_b32_d512_k4 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k4.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d512_k4 \
        > logs/ff_b32_d512_k4.log
}
function ff_b32_d512_k5 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/ff-d512-k5.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save ff_b32_d512_k5 \
        > logs/ff_b32_d512_k5.log
}
# Hmm sweep
function hmm_b4_d256_k128 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k128.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k128 \
        > logs/hmm_b4_d256_k128.log
}
function hmm_b4_d256_k256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k256.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k256 \
        > logs/hmm_b4_d256_k256.log
}
function hmm_b4_d512_k128 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k128.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d512_k128 \
        > logs/hmm_b4_d512_k128.log
}
function hmm_b4_d512_k256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k256.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d512_k256 \
        > logs/hmm_b4_d512_k256.log
}
function hmm_b32_d256_k128 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k128.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b32_d256_k128 \
        > logs/hmm_b32_d256_k128.log
}
function hmm_b32_d256_k256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k256.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b32_d256_k256 \
        > logs/hmm_b32_d256_k256.log
}
function hmm_b32_d512_k128 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k128.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b32_d512_k128 \
        > logs/hmm_b32_d512_k128.log
}
function hmm_b32_d512_k256 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k256.yaml \
        --bsz 32 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b32_d512_k256 \
        > logs/hmm_b32_d512_k256.log
}
# Hmm overfitting script
function hmm_b4_d256_k128_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k128.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d256_k128_overfit.log
}
function hmm_b4_d256_k256_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k256.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d256_k256_overfit.log
}
function hmm_b4_d256_k512_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d256_k512_overfit.log
}
function hmm_b4_d512_k128_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k128.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d512_k128_overfit.log
}
function hmm_b4_d512_k256_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k256.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d512_k256_overfit.log
}
function hmm_b4_d512_k512_overfit {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k512.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 2 \
        --overfit \
        > overfit_logs/hmm_b4_d512_k512_overfit.log
}
# Hmm tvm script
function hmm_b4_d256_k128_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k128.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k128_tvm \
        > tvm_logs/hmm_b4_d256_k128_tvm.log
}
function hmm_b4_d256_k256_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k256.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k256_tvm \
        > tvm_logs/hmm_b4_d256_k256_tvm.log
}
function hmm_b4_d256_k512_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k512_tvm \
        > tvm_logs/hmm_b4_d256_k512_tvm.log
}
function hmm_b4_d256_k1024_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k1024.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k1024_tvm \
        > tvm_logs/hmm_b4_d256_k1024_tvm.log
}
function hmm_b4_d512_k128_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k128.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k128_tvm \
        > tvm_logs/hmm_b4_d512_k128_tvm.log
}
function hmm_b4_d512_k256_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k256.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k256_tvm \
        > tvm_logs/hmm_b4_d512_k256_tvm.log
}
function hmm_b4_d512_k512_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k512.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k512_tvm \
        > tvm_logs/hmm_b4_d512_k512_tvm.log
}
function hmm_b4_d512_k1024_tvm {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k1024.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k1024_tvm \
        > tvm_logs/hmm_b4_d512_k1024_tvm.log
}
# Hmm seed sweep
function hmm_b4_d256_k512_s1234 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k512_s1234 \
        --seed 1234 \
        > logs/hmm_b4_d256_k512_s1234.log
}
function hmm_b4_d256_k512_s1 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k512_s1 \
        --seed 1 \
        > logs/hmm_b4_d256_k512_s1.log
}
function hmm_b4_d256_k512_s2357 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k512_s2357 \
        --seed 2357 \
        > logs/hmm_b4_d256_k512_s2357.log
}
function hmm_b4_d256_k512_s2468 {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512.yaml \
        --bsz 4 \
        --num-epochs 100 \
        --patience 8 \
        --save hmm_b4_d256_k512_s2468 \
        --seed 2468 \
        > logs/hmm_b4_d256_k512_s2468.log
}
# Hmm tvm oldres script
function hmm_b4_d256_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k128_oldres \
        > tvm_logs/hmm_b4_d256_k128_oldres.log
}
function hmm_b4_d256_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k256_oldres \
        > tvm_logs/hmm_b4_d256_k256_oldres.log
}
function hmm_b4_d256_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k512_oldres \
        > tvm_logs/hmm_b4_d256_k512_oldres.log
}
function hmm_b4_d256_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d256-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d256_k1024_oldres \
        > tvm_logs/hmm_b4_d256_k1024_oldres.log
}
function hmm_b4_d512_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k128_oldres \
        > tvm_logs/hmm_b4_d512_k128_oldres.log
}
function hmm_b4_d512_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k256_oldres \
        > tvm_logs/hmm_b4_d512_k256_oldres.log
}
function hmm_b4_d512_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k512_oldres \
        > tvm_logs/hmm_b4_d512_k512_oldres.log
}
function hmm_b4_d512_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/hmm-d512-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save hmm_b4_d512_k1024_oldres \
        > tvm_logs/hmm_b4_d512_k1024_oldres.log
}
# PoeHmm tvm oldres script
function poehmm_b4_d256_n2_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n2-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n2_k128_oldres \
        > tvm_logs/poehmm_b4_d256_n2_k128_oldres.log
}
function poehmm_b4_d256_n2_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n2-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n2_k256_oldres \
        > tvm_logs/poehmm_b4_d256_n2_k256_oldres.log
}
function poehmm_b4_d256_n2_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n2-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n2_k512_oldres \
        > tvm_logs/poehmm_b4_d256_n2_k512_oldres.log
}
function poehmm_b4_d256_n2_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n2-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n2_k1024_oldres \
        > tvm_logs/poehmm_b4_d256_n2_k1024_oldres.log
}
function poehmm_b4_d256_n3_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n3-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n3_k128_oldres \
        > tvm_logs/poehmm_b4_d256_n3_k128_oldres.log
}
function poehmm_b4_d256_n3_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n3-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n3_k256_oldres \
        > tvm_logs/poehmm_b4_d256_n3_k256_oldres.log
}
function poehmm_b4_d256_n3_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n3-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n3_k512_oldres \
        > tvm_logs/poehmm_b4_d256_n3_k512_oldres.log
}
function poehmm_b4_d256_n3_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n3-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n3_k1024_oldres \
        > tvm_logs/poehmm_b4_d256_n3_k1024_oldres.log
}
function poehmm_b4_d256_n4_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n4-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n4_k128_oldres \
        > tvm_logs/poehmm_b4_d256_n4_k128_oldres.log
}
function poehmm_b4_d256_n4_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n4-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n4_k256_oldres \
        > tvm_logs/poehmm_b4_d256_n4_k256_oldres.log
}
function poehmm_b4_d256_n4_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n4-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n4_k512_oldres \
        > tvm_logs/poehmm_b4_d256_n4_k512_oldres.log
}
function poehmm_b4_d256_n4_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n4-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n4_k1024_oldres \
        > tvm_logs/poehmm_b4_d256_n4_k1024_oldres.log
}
function poehmm_b4_d256_n5_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n5-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n5_k128_oldres \
        > tvm_logs/poehmm_b4_d256_n5_k128_oldres.log
}
function poehmm_b4_d256_n5_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n5-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n5_k256_oldres \
        > tvm_logs/poehmm_b4_d256_n5_k256_oldres.log
}
function poehmm_b4_d256_n5_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n5-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n5_k512_oldres \
        > tvm_logs/poehmm_b4_d256_n5_k512_oldres.log
}
function poehmm_b4_d256_n5_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d256-n5-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d256_n5_k1024_oldres \
        > tvm_logs/poehmm_b4_d256_n5_k1024_oldres.log
}
function poehmm_b4_d512_n2_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n2-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n2_k128_oldres \
        > tvm_logs/poehmm_b4_d512_n2_k128_oldres.log
}
function poehmm_b4_d512_n2_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n2-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n2_k256_oldres \
        > tvm_logs/poehmm_b4_d512_n2_k256_oldres.log
}
function poehmm_b4_d512_n2_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n2-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n2_k512_oldres \
        > tvm_logs/poehmm_b4_d512_n2_k512_oldres.log
}
function poehmm_b4_d512_n2_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n2-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n2_k1024_oldres \
        > tvm_logs/poehmm_b4_d512_n2_k1024_oldres.log
}
function poehmm_b4_d512_n3_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n3-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n3_k128_oldres \
        > tvm_logs/poehmm_b4_d512_n3_k128_oldres.log
}
function poehmm_b4_d512_n3_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n3-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n3_k256_oldres \
        > tvm_logs/poehmm_b4_d512_n3_k256_oldres.log
}
function poehmm_b4_d512_n3_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n3-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n3_k512_oldres \
        > tvm_logs/poehmm_b4_d512_n3_k512_oldres.log
}
function poehmm_b4_d512_n3_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n3-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n3_k1024_oldres \
        > tvm_logs/poehmm_b4_d512_n3_k1024_oldres.log
}
function poehmm_b4_d512_n4_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n4-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n4_k128_oldres \
        > tvm_logs/poehmm_b4_d512_n4_k128_oldres.log
}
function poehmm_b4_d512_n4_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n4-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n4_k256_oldres \
        > tvm_logs/poehmm_b4_d512_n4_k256_oldres.log
}
function poehmm_b4_d512_n4_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n4-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n4_k512_oldres \
        > tvm_logs/poehmm_b4_d512_n4_k512_oldres.log
}
function poehmm_b4_d512_n4_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n4-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n4_k1024_oldres \
        > tvm_logs/poehmm_b4_d512_n4_k1024_oldres.log
}
function poehmm_b4_d512_n5_k128_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n5-k128-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n5_k128_oldres \
        > tvm_logs/poehmm_b4_d512_n5_k128_oldres.log
}
function poehmm_b4_d512_n5_k256_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n5-k256-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n5_k256_oldres \
        > tvm_logs/poehmm_b4_d512_n5_k256_oldres.log
}
function poehmm_b4_d512_n5_k512_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n5-k512-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n5_k512_oldres \
        > tvm_logs/poehmm_b4_d512_n5_k512_oldres.log
}
function poehmm_b4_d512_n5_k1024_oldres {
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --devid 0 \
        --model-config configs/poehmm-d512-n5-k1024-oldres.yaml \
        --num-epochs 100 \
        --bsz 4 \
        --patience 8 \
        --save poehmm_b4_d512_n5_k1024_oldres \
        > tvm_logs/poehmm_b4_d512_n5_k1024_oldres.log
}

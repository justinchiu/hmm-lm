#!/bin/bash
#
l_cluster=/home/jtc257/cpp/brown-cluster/wcluster

# viterbi output brown
#input=viterbi_output/ptb_bucket_mshmm_k32768_wps512_spw256_tspw128_ed256_d256_dp0_tdp0.5_cdp1_sdp0_dtnone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc0.viterbi.txt
#$l_cluster --text $input --c 45 --output_dir viterbi_clusters/mshmm_k32768_nb128

# already ran these
#$l_cluster --text .data/PTB/ptb.txt --c 45 --output_dir viterbi_clusters/brown_45
#$l_cluster --text .data/PTB/ptb.digits.txt.test --c 45 --output_dir viterbi_clusters/brown_digits_45
#$l_cluster --text .data/penn-treebank/ptb.train.txt --c 45 --output_dir viterbi_clusters/brown_mik_45

# viterbi output
input=viterbi_output/wsj_bucket_mshmm_k16384_wps512_spw128_tspw64_ed256_d256_cd16_dp0_tdp0.5_cdp1_sdp0_dtnone_wd0_tokens_b512_adamw_lr0.01_c5_tw_nas0_pw1_asbrown_nb128_nc0_ncs0_spc0_n5_r0_ns0_fc0_eword_ednone_nh0_sind.viterbi.txt
$l_cluster --text $input --c 45 --output_dir viterbi_clusters/wsj_mshmm_k16384_nb128

$l_cluster --text .data/wsj/wsj.raw --c 45 --output_dir viterbi_clusters/brown_wsjraw_45
$l_cluster --text .data/wsj/wsj.txt --c 45 --output_dir viterbi_clusters/brown_wsj_45

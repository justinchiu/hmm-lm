
from itertools import product

import os

def make_script(num_states, num_features, dropout, feat_dropout):
    header = f"""#!/bin/bash
#SBATCH -J dp{num_states}{num_features}{dropout}{feat_dropout}
#SBATCH -p rush
#SBATCH --nodelist=rush-compute01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks 1
#SBATCH --mem 12G

source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh

hmmenv
cd /home/jtc257/python/hmm-lm

python main.py --lr 0.001 --column_dropout 0 \
    --transition_dropout {dropout} --feature_dropout {feat_dropout}\
    --dropout_type state \
    --model blhmm --bsz 128 --num_classes {num_states} \
    --emb_dim 256 --hidden_dim 256 \
    --dataset ptb --iterator bucket --parameterization smp \
    --projection_method static --update_proj 1 \
    --num_features {num_features} \
    --anti 0 --l2norm 0 --sm_emit 1 \
    --eff 1
"""
    return header

def make_filename(num_states, num_features, dropout, feat_dropout):
    filename = f"dp-s{num_states}-f{num_features}-{dropout}-{feat_dropout}.sub"
    return filename

#grid_num_states = [8192, 4096, 2048, 1024, 512]
#grid_num_features = [1024, 512, 256, 128, 64]
#grid_num_states = [16384, 8192]
#grid_num_features = [16384, 8192, 4096]
#grid_num_states = [8192]
#grid_num_features = [4096, 2048, 1024]
grid_num_states = [4096]
grid_num_features = [2048, 1024, 512]

grid_dropout_type = ["state"]
#grid_dropout = [0, 0.3]
grid_dropout = [0, 0.1, 0.2]
grid_feat_dropout = [0, 0.1, 0.2]

for num_states, num_features, dropout, feat_dropout in product(
    grid_num_states,
    grid_num_features,
    grid_dropout,
    grid_feat_dropout,
):
    filename = make_filename(num_states, num_features, dropout, feat_dropout)
    body = make_script(num_states, num_features, dropout, feat_dropout)
    # write script content
    with open(filename, "w") as f:
        f.write(body)
    # execute script
    os.system(f"sbatch {filename}")

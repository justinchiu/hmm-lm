
from itertools import product

import os

def make_script(num_states, dropout_type, dropout):
    header = f"""#!/bin/bash
#SBATCH -J dps{num_states}{dropout_type}{dropout}
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
    --transition_dropout {dropout} --dropout_type {dropout_type} \
    --model blhmm --bsz 128 --num_classes {num_states} \
    --emb_dim 256 --hidden_dim 256 \
    --dataset ptb --iterator bucket --parameterization softmax 
"""
    return header

def make_filename(num_states, dropout_type, dropout):
    filename = f"sdp-s{num_states}-{dropout_type}{dropout}.sub"
    return filename

#grid_num_states = [8192, 4096, 2048, 1024, 512]
#grid_num_features = [1024, 512, 256, 128, 64]
grid_num_states = [16384, 8192]

grid_dropout_type = ["state"]
grid_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

for num_states, dropout_type, dropout in product(
    grid_num_states,
    grid_dropout_type, grid_dropout,
):
    filename = make_filename(num_states, dropout_type, dropout)
    body = make_script(num_states, dropout_type, dropout)
    # write script content
    with open(filename, "w") as f:
        f.write(body)
    # execute script
    os.system(f"sbatch {filename}")
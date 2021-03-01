
from itertools import product

import os

def make_script(num_states):
    header = f"""#!/bin/bash
#SBATCH -J b{num_states}
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

python main.py  --lr 0.001 --column_dropout 1 --transition_dropout 0 --dropout_type none \
    --model blhmm --bsz 256 --num_classes {num_states} --emb_dim 256 \
    --hidden_dim 256 --dataset ptb --iterator bucket
"""
    return header

def make_filename(num_states):
    filename = f"softmaxhmm-s{num_states}.sub"
    return filename

grid_num_states = [16384, 8192, 4096, 2048, 1024, 512]
#grid_num_states = [16384]

for num_states in grid_num_states:
    filename = make_filename(num_states)
    body = make_script(num_states)
    # write script content
    with open(filename, "w") as f:
        f.write(body)
    # execute script
    os.system(f"sbatch {filename}")

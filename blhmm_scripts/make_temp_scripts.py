
from itertools import product

import os

def make_script(num_states, num_features, learn_temp):
    header = f"""#!/bin/bash
#SBATCH -J b{num_states}f{num_features}
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

python main.py --lr 0.001 --column_dropout 0 --transition_dropout 0 --dropout_type none \
    --model blhmm --bsz 256 --num_classes {num_states} \
    --emb_dim 256 --hidden_dim 256 \
    --dataset ptb --iterator bucket --parameterization smp \
    --projection_method static --update_proj 1 \
    --num_features {num_features} \
    --anti 0 --l2norm 0 --sm_emit 1 \
    --eff 1 --learn_temp {learn_temp}
"""
    return header

def make_filename(num_states, num_features, learn_temp):
    filename = f"t{learn_temp}-{num_states}-{num_features}.sub"
    return filename

#grid_num_states = [8192, 4096, 2048, 1024, 512]
#grid_num_features = [1024, 512, 256, 128, 64]
#grid_num_states = [16384, 8192]
#grid_num_features = [8192]
grid_num_states = [16384, 8192, 4096, 2048, 1024, 512]
grid_num_features = [2048, 1024, 512, 256]

configurations = [
    (4096, 1024, "mul"),
    (4096, 512, "mul"),
    (4096, 1024, "add"),
    (4096, 512, "add"),
    #(4096, 1024, "none"),
    #(4096, 512, "none"),
]

#for num_states, num_features in product(grid_num_states, grid_num_features):
for num_states, num_features, learn_temp in configurations:
    filename = make_filename(num_states, num_features, learn_temp)
    body = make_script(num_states, num_features, learn_temp)
    # write script content
    with open(filename, "w") as f:
        f.write(body)
    # execute script
    os.system(f"sbatch {filename}")

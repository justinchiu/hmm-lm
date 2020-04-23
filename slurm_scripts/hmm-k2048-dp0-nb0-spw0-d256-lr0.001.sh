#!/bin/bash
#SBATCH -J hmm-k2048-dp0-d256-lr0.001                         # Job name
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 2                                # Total number of cores requested
#SBATCH --mem=4G                          # Total amount of (real) memory requested (per node)
#SBATCH -t 48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=rush                     # Request partition for resource allocation
#SBATCH --nodelist=rush-compute01
#SBATCH --gres=gpu:1                         # Specify a list of generic consumable resources (per node)

source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh
py14env
python main.py --lr 0.001 --column_dropout 1 --transition_dropout 0 --dropout_type none --model hmm --bsz 128 --num_classes 2048 --emb_dim 256 --hidden_dim 256


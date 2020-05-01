#!/bin/bash
#SBATCH -J mshmm-k32768-dp0.5-nb32-spw1024-d256-lr0.001                         # Job name
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 16                                # Total number of cores requested
#SBATCH --mem=24G                          # Total amount of (real) memory requested (per node)
#SBATCH -t 48:00:00                          # Time limit (hh:mm:ss)
#SBATCH --partition=rush                     # Request partition for resource allocation
#SBATCH --nodelist=rush-compute01
#SBATCH --gres=gpu:1                         # Specify a list of generic consumable resources (per node)

source /home/jtc257/.bashrc
source /home/jtc257/scripts/env.sh
py14env
python main.py --lr 0.001 --column_dropout 1 --transition_dropout 0.5 --model mshmm --assignment brown --states_per_word 512 --num_clusters 128 --num_classes 65536 --bsz 512 --eval_bsz 5096
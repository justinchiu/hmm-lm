
hmm256 () {
    WANDB_MODE=dryrun python main.py --lr 0.001 --column_dropout 1 --transition_dropout 0 --dropout_type none --model hmm --bsz 128 --num_classes 512 --emb_dim 256 --hidden_dim 256 --dataset ptb --iterator bucket
}



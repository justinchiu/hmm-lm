
from argparse import ArgumentParser
import torch_struct as ts

def get_args():
    parser = ArgumentParser()
    # learning args
    parser.add_argument("--bsz", default=1024, type=int,)
    parser.add_argument("--eval_bsz", default=1024, type=int,)
    parser.add_argument("--bsz_fn", choices=["tokens", "sentences"], default="tokens",)
    parser.add_argument("--lr", default=1e-3, type=float,)
    parser.add_argument("--clip", default=5, type=float,)
    parser.add_argument("--beta1", default=0.9, type=float,)
    parser.add_argument("--beta2", default=0.999, type=float,)
    parser.add_argument("--wd", default=0.000, type=float,)
    parser.add_argument("--decay", default=4, type=float,)
    parser.add_argument("--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",)
    parser.add_argument("--schedule",
        choices=["reducelronplateau", "noam"],
        default="reducelronplateau",)
    parser.add_argument("--warmup_steps", default=500, type=int,)
    parser.add_argument("--patience", default=4, type=int,)
    parser.add_argument("--bptt", default=35, type=int, )
    parser.add_argument("--num_epochs", default=50, type=int,)
    parser.add_argument("--num_checks", default=4, type=int,)
    parser.add_argument("--report_every", default=5000,)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--devid", default=0, type=int,)
    parser.add_argument("--aux_devid", default=1, type=int,)
    parser.add_argument("--model", choices=[
        "shmm", "dhmm", "chmm", "hmm", "lstm", "ff",
        "mshmm",
    ], default="chmm",)
    parser.add_argument("--seed", default=1111, type=int,)
    parser.add_argument("--eval_only", default="",)

    add_nn_args(parser)

    # add chmm args
    add_chmm_args(parser)
    add_dhmm_args(parser)

    return parser.parse_args()

def add_nn_args(parser):
    parser.add_argument_group("nn")
    parser.add_argument("--tie_weights", default=0, type=int,)
    parser.add_argument("--ngrams", default=5, type=int,)

def add_chmm_args(parser):
    parser.add_argument_group("chmm")
    parser.add_argument("--num_layers", default=1, type=int,)
    parser.add_argument("--emb_dim", default=256, type=int,)
    parser.add_argument("--hidden_dim", default=256, type=int,)
    parser.add_argument("--dropout", default=0, type=float,)
    parser.add_argument("--num_classes", default=16384, type=int,)
    parser.add_argument("--words_per_state", default=512, type=int,)
    parser.add_argument("--states_per_word", default=128, type=int,)
    parser.add_argument("--ffnn", default="", type=str, help="default is oldres")
    parser.add_argument("--tw", default="", type=str, help="default is no weight tying")
    parser.add_argument("--transition_dropout", default=0, type=float,)
    parser.add_argument("--column_dropout", default=0, type=int, help="0 = no coldrop")
    parser.add_argument("--start_dropout", default=0, type=float,)
    parser.add_argument("--dropout_type", choices=[
        "transition", "column", "startcolumn", "state", "cluster",
    ])
    parser.add_argument("--assignment", choices=[
        "brown", "unevenbrown",
        "word2vec",
        "uniform",
    ], default="brown",)
    parser.add_argument("--num_clusters", default=128, type=int, help="number of brown clusters")
    parser.add_argument("--num_common", default=0, type=int, help="top k common words (only unevenbrown)")
    parser.add_argument("--num_common_states", default=0, type=int, help="number of common states (only unevenbrown)")
    parser.add_argument("--states_per_common", default=0, type=int, help="repeat each common word (only unevenbrown)")

def add_dhmm_args(parser):
    parser.add_argument_group("dhmm")
    parser.add_argument("--assignment_noise", choices=["gumbel", "none"], default="gumbel",)
    parser.add_argument("--noise_anneal_steps", default=0, type=int, help="0 = no anneal")
    parser.add_argument("--keep_counts", default=0, type=int,)
    parser.add_argument("--log_counts", default=0, type=int)
    parser.add_argument("--count_posterior", choices=[], default="",)
    parser.add_argument("--posterior_weight", type=float, default=1,)

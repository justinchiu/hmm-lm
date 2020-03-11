# hmm-lm

# Data
The data is Penn Treebank, where each example is a single sentence
as opposed to a continuous stream of text.
<eos> is not appended to sentences and therefore not included in the likelihood.
We use `torchtext` for pretty much everything, see `datasets/ptb.py` for the details.

Likelihood (and perplexity) is computed at the token level.

# Experiments
To regenerate `scripts/experiments.sh`, run `python scripts/gen_scripts.py > scripts/experiments.sh`.

CHMM:
```
CUDA_VISIBLE_DEVICES=1 python -u main.py --model-config configs/chmm-d256-k16384-wps512-spw128-oldres.yaml --bsz 4 --num-epochs 100 --patience 8 --devid 0 --save
chmm_d256_k16384_wps512_spw128_oldres > tvm_logs/chmm_d256_k16384_wps512_spw128_oldres.log
```

Use the following commands to train models:
```
source scripts/run_experiments && run_lstms
source scripts/run_experiments && run_ffs
source scripts/run_experiments && run_hmms
```

For some overfitting experiments, run:
```
source scripts/run_experiments && run_lstms_overfit
source scripts/run_experiments && run_ffs_overfit
source scripts/run_experiments && run_hmms_overfit
```

(DEPRECATED, need to revisit) K-means clustering of LSTM hidden states
```
python kmeans.py > kmeans.log
```

# Results

## Current results

Unless indicated otherwise,
all models have tied input and output embeddings (if there is freedom).
FF and LSTM models have two layers, with dropout = 0.3 for dim = 256 
and dropout = 0.5 for dim = 512.
Training is terminated at 100 epochs.

|x| Model | bsz | dim | k    | Train PPL | Valid PPL | Log                      |
|-| ----- | --- | --- | ---- | --------- | --------- | ------------------------ |
| | FF    | 32  | 256 | 5    | 152.35    | 161.70    | logs/ff_b32_d256_k5.log  |
|x| FF    | 4   | 256 | 5    | 327.58    | 304.89    | logs/ff_b4_d256_k5.log   |
|x| FF    | 32  | 512 | 5    | 169.04    | 181.61    | logs/ff_b32_d512_k5.log  |
|x| LSTM  | 32  | 256 |      | 63.46     | 97.61     | logs/lstm_b32_d256.log   |
|x| LSTM  | 4   | 256 |      |           |           | logs/lstm_b4_d256.log    |
|x| LSTM  | 32  | 512 |      | 70.35     | 93.70     | logs/lstm_b32_d512.log   |
|x| LSTM  | 4   | 512 |      |           |           | logs/lstm_b4_d512.log    |
| | HMM   | 4   | 256 | 128  | 193.89    | 226.33    | staging_logs/hmm_b4_d256_k128_oldres.log  |
| | HMM   | 4   | 256 | 256  | 173.72    | 208.60    | staging_logs/hmm_b4_d256_k256_oldres.log  |
| | HMM   | 4   | 256 | 512  | 140.08    | 181.19    | staging_logs/hmm_b4_256_k512_oldres.log   |
| | HMM   | 4   | 256 | 1024 | 127.80    | 174.47    | staging_logs/hmm_b4_d256_k1024_oldres.log |
| | HMM   | 4   | 512 | 128  | 197.41    | 227.19    | tvm_logs/hmm_b4_d512_k128_oldres.log  |
| | HMM   | 4   | 512 | 256  | 163.83    | 200.59    | tvm_logs/hmm_b4_d512_k256_oldres.log  |
| | HMM   | 4   | 512 | 512  | 139.54    | 182.69    | tvm_logs/hmm_b4_512_k512_oldres.log   |
| | HMM   | 4   | 512 | 1024 | 124.45    | 172.17    | tvm_logs/hmm_b4_d512_k1024_oldres.log |

Could probably bring these down a bit with more tuning.
|x| Model | bsz | dim | k   | Train PPL | Valid PPL | Log                  |
|-| ----- | --- | --- | --- | --------- | --------- | -------------------- |
| | FF    | 32  | 256 | 5   | 152.35    | 161.70    | logs/ff_b32_d256_k5.log   |
| | FF    | 32  | 256 | 4   | 149.23    | 160.92    | logs/ff_b32_d256_k4.log   |
| | FF    | 32  | 256 | 3   | 149.61    | 164.68    | logs/ff_b32_d256_k3.log   |
| | FF    | 32  | 256 | 2   | 178.90    | 202.50    | logs/ff_b32_d256_k2.log   |


Overfitting experiments

|x| Model | bsz | dim | k   | Train PPL | Valid PPL | Log                          |
|-| ----- | --- | --- | --- | --------- | --------- | ---------------------------- |
| | FF    | 32  | 256 |     |           |           | ff_b32_d256_overfit.log      |
| | FF    | 4   | 256 |     |           |           | ff_b4_d256_overfit.log       |
| | FF    | 32  | 512 |     |           |           | ff_b32_d512_overfit.log      |
| | FF    | 4   | 512 |     |           |           | ff_b4_d512_overfit.log       |
| | LSTM  | 32  | 256 |     |           |           | lstm_b32_d256_overfit.log    |
| | LSTM  | 4   | 256 |     |           |           | lstm_b4_d256_overfit.log     |
| | LSTM  | 32  | 512 |     |           |           | lstm_b32_d512_overfit.log    |
| | LSTM  | 4   | 512 |     |           |           | lstm_b4_d512_overfit.log     |
| | HMM   | 4   | 256 | 128 |           |           | hmm_b4_d256_k128_overfit.log |
| | HMM   | 4   | 512 | 128 |           |           | hmm_b4_d512_k128_overfit.log |
| | HMM   | 4   | 256 | 256 |           |           | hmm_b4_d256_k256_overfit.log |
| | HMM   | 4   | 512 | 256 |           |           | hmm_b4_d512_k256_overfit.log |

NOTE: alternative layer norm location in residual block

|x| Model | bsz | dim | k   | Train PPL | Valid PPL | Log                  |
|-| ----- | --- | --- | --- | --------- | --------- | -------------------- |
| | HMM   | 4   | 256 | 128 | 204.02    | 247.64    | tvm_logs/hmm_b4_d256_k128_tvm.log |
| | HMM   | 4   | 256 | 256 | 182.98    | 229.97    | tvm_logs/hmm_b4_d256_k256_tvm.log |
| | HMM   | 4   | 256 | 512 | 172.52    | 223.48    | tvm_logs/hmm_b4_d256_k512_tvm.log |
| | HMM   | 4   | 256 | 1024 |     |     | tvm_logs/hmm_b4_d256_k1024_tvm.log |

NOTE: Old experiments without EOS (for reference)

| Model               | Train PPL | Valid PPL | Log             |
| ------------------- | --------- | --------- | --------------- |
| FF   bsz 32         | 178.99    | 184.39    | ff_b32.log      |
| FF   bsz 4          | X         | X         | ff_b4.log       |
| LSTM bsz 32         | 86.10     | 111.19    | lstm_b32.log    |
| LSTM bsz 4          | 104.84    | 124.25    | lstm_b4.log     |
| HMM  bsz 4  K = 128 | 216.59    | 253.60    | hmm_b4_k256.log |
| HMM  bsz 4  K = 256 | 180.84    | 220.93    | hmm_b4_k256.log |
| HMM  bsz 4  K = 512 | 170.40    | 202.94    | preliminary     |

## Results from [Buys et. al.](https://openreview.net/forum?id=rJxEso0osm)

Results taken from the paper.
The LSTM uses batch size is 32, hidden dimension ~ 900 (10M parameters), 
No mention of dropout.


| Model | Valid PPL |
| ----- | --------- |
| LSTM  | 80.61     |
| HMM   | 284.59    |

With a similar model, run with the command
```
python main.py --devid 1 --model-config configs/jan_lstm.yaml --bsz 32 > jan_lstm.log
```
| Model | Valid PPL |
| ----- | --------- |
| LSTM  | 157.27    |


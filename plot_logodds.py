import numpy as np

import chart_studio.plotly as py
import plotly.graph_objects as go

probs_and_counts_ff = np.load("probs_and_counts_ff.npy")
"""
next_probs_and_counts_ff = np.load("nextprobs_and_counts_ff.npy")
probs_and_counts_lstm = np.load("probs_and_counts_lstm.npy")
next_probs_and_counts_lstm = np.load("nextprobs_and_counts_lstm.npy")
"""

# buckets of 10 up to 50
W = 10
N = 50
import pdb; pdb.set_trace()
fig = go.Figure()
for A in range(0, N, W):
    fig.add_trace(go.Box(
        y = (probs_and_counts_ff[1] / probs_and_counts_ff[0])[
            (A * W <= probs_and_counts_ff[0]) * (probs_and_counts_ff[0] < (A+1) * W)
        ],
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div ff prob)", sharing="public", auto_open=False,
)

doplot = False
if doplot:
    py.plot(
        [
            go.Scatter(
                y=probs_and_counts_ff[1] / probs_and_counts_ff[0],
                x=probs_and_counts_ff[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log hmm prob div ff prob)", sharing="public", auto_open=False,
    )
    py.plot(
        [
            go.Scatter(
                y=nextprobs_and_counts_ff[1] / nextprobs_and_counts_ff[0],
                x=nextprobs_and_counts_ff[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log next hmm prob div ff prob) div counts", sharing="public", auto_open=False,
    )

    py.plot(
        [
            go.Scatter(
                y=probs_and_counts_lstm[1] / probs_and_counts_lstm[0],
                x=probs_and_counts_lstm[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log hmm prob div lstm prob)", sharing="public", auto_open=False,
    )
    py.plot(
        [
            go.Scatter(
                y=nextprobs_and_counts_lstm[1] / nextprobs_and_counts_lstm[0],
                x=nextprobs_and_counts_lstm[0],
                mode="markers",
            ),
        ],
        filename="counts vs mean(log next hmm prob div lstm prob) div counts", sharing="public", auto_open=False,
    )

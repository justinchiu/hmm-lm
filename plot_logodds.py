import numpy as np

import chart_studio.plotly as py
import plotly.graph_objects as go

probs_and_counts_ff = np.load("probs_and_counts_ff.npy")
probs_and_counts_lstm = np.load("probs_and_counts_lstm.npy")
"""
next_probs_and_counts_ff = np.load("nextprobs_and_counts_ff.npy")
probs_and_counts_lstm = np.load("probs_and_counts_lstm.npy")
next_probs_and_counts_lstm = np.load("nextprobs_and_counts_lstm.npy")
"""

import pdb; pdb.set_trace()

# buckets of 5 up to 50
W = 2
N = 50

fig = go.Figure(
    layout_title_text = "Counts vs log odds ratio hmm and ff",
)
for A in range(0, N, W):
    fig.add_trace(go.Box(
        y = (probs_and_counts_ff[1] / probs_and_counts_ff[0])[
            (A <= probs_and_counts_ff[0]) * (probs_and_counts_ff[0] < A + W)
        ],
        boxpoints = "all",
        name = f"Counts [{A}, {A + W})",
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div ff prob)", sharing="public", auto_open=False,
)

fig = go.Figure(
    layout_title_text = "Counts vs log odds ratio hmm and lstm",
)
for A in range(0, N, W):
    fig.add_trace(go.Box(
        y = (probs_and_counts_lstm[1] / probs_and_counts_lstm[0])[
            (A <= probs_and_counts_lstm[0]) * (probs_and_counts_lstm[0] < A + W)
        ],
        boxpoints = "all",
        name = f"Counts [{A}, {A + W})",
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div lstm prob)", sharing="public", auto_open=False,
)

fig = go.Figure(
    layout_title_text = "Counts vs log odds ratio next word hmm and ff",
)
for A in range(0, N, W):
    fig.add_trace(go.Box(
        y = (probs_and_counts_ff[1] / probs_and_counts_ff[0])[
            (A <= probs_and_counts_ff[0]) * (probs_and_counts_ff[0] < A + W)
        ],
        boxpoints = "all",
        name = f"Counts [{A}, {A + W})",
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div ff prob next word)", sharing="public", auto_open=False,
)

fig = go.Figure(
    layout_title_text = "Counts vs log odds ratio next word hmm and lstm",
)
for A in range(0, N, W):
    fig.add_trace(go.Box(
        y = (probs_and_counts_lstm[1] / probs_and_counts_lstm[0])[
            (A <= probs_and_counts_lstm[0]) * (probs_and_counts_lstm[0] < A + W)
        ],
        boxpoints = "all",
        name = f"Counts [{A}, {A + W})",
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div lstm prob next word)", sharing="public", auto_open=False,
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

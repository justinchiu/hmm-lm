import numpy as np

import chart_studio.plotly as py
import plotly.graph_objects as go

probs_and_counts_ff = np.load("probs_and_counts_ff.npy")
probs_and_counts_lstm = np.load("probs_and_counts_lstm.npy")
nextprobs_and_counts_ff = np.load("nextprobs_and_counts_ff.npy")
nextprobs_and_counts_lstm = np.load("nextprobs_and_counts_lstm.npy")
probs_and_pos_ff = np.load("probs_and_pos_ff.npy")
probs_and_pos_lstm = np.load("probs_and_pos_lstm.npy")

import pdb; pdb.set_trace()

# log odds and word counts
# buckets of 5 up to 50
W = 2
N = 50

fig = go.Figure(
    layout_title_text = "Counts vs log odds ratio hmm and ff",
)
for A in range(1, N, W):
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
for A in range(1, N, W):
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
for A in range(1, N, W):
    fig.add_trace(go.Box(
        y = (nextprobs_and_counts_ff[1] / nextprobs_and_counts_ff[0])[
            (A <= nextprobs_and_counts_ff[0]) * (nextprobs_and_counts_ff[0] < A + W)
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
for A in range(1, N, W):
    fig.add_trace(go.Box(
        y = (nextprobs_and_counts_lstm[1] / nextprobs_and_counts_lstm[0])[
            (A <= nextprobs_and_counts_lstm[0]) * (nextprobs_and_counts_lstm[0] < A + W)
        ],
        boxpoints = "all",
        name = f"Counts [{A}, {A + W})",
    ))
py.plot(
    fig,
    filename="box plot counts vs mean(log hmm prob div lstm prob next word)",
    sharing="public", auto_open=False,
)

# log odds and position in sentence
fig = go.Figure(
)
fig.add_trace(go.Scatter(
    x = np.arange(probs_and_pos_ff.shape[1]),
    y = probs_and_pos_ff[1] / probs_and_pos_ff[0],
    name = "log hmm / ff",
))
fig.add_trace(go.Scatter(
    x = np.arange(probs_and_pos_lstm.shape[1]),
    y = probs_and_pos_lstm[1] / probs_and_pos_lstm[0],
    name = "log hmm / lstm",
))
fig.update_layout(
    title = "Position vs mean(log odds)",
    xaxis_title = "position in sentence",
    yaxis_title = "average log odds of current word",
)
py.plot(
    fig,
    filename="plot pos vs mean(log odds)",
    sharing="public", auto_open=False,
)

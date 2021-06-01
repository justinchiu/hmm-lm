import math

import pathlib

import numpy as np
from numpy.linalg import svd

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", font_scale=1.5)

dir = pathlib.Path("svd")

for path in dir.glob("*.npy"):
    singular_values = np.load(path)
    #sns.scatterplot(
    g = sns.relplot(
        data=singular_values,
        kind="line",
        linewidth=2,
        aspect=1.3,
    )
    g.set_axis_labels("Index", "Singular values")
    g.tight_layout()
    graphpath = str(path.with_suffix(".png"))
    g.savefig(graphpath)
    plt.close()

    print("Processed:")
    print(path)
    print("Saved to: {graphpath}")
    print("Num singular values > 1e-5: {(singular_values > 1e-5).sum()}")

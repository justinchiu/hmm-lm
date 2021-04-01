
import numpy as np

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

def plot_svs(transition, filename):
    # log
    svs = np.linalg.svd(np.exp(transition), compute_uv=False)
    fig, ax = plt.subplots()
    g = sns.scatterplot(x=np.arange(len(svs)),y=svs, ax=ax)
    fig.savefig(f"{filename}-svd.png")
    plt.close(fig)

transition_dir = Path("transitions")
for filename in transition_dir.iterdir():
    # log
    transition = np.load(filename)
    print(filename)
    plot_svs(transition, filename)

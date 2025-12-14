import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from data_loader import load_dataset
from evaluator import Evaluator
from algorithms import run_ga
import os

PARAMS_GA = {'pop_size': 110, 'n_gen': 20, 'p_cruce': 0.88, 'p_mutacion': 0.12, 'tam_torneo': 4}  # GA parameters

DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']  # datasets

CARPETA_IMG = "images"
os.makedirs(CARPETA_IMG, exist_ok=True)  # create folder if doesn't exist

# prepare grid figure
n_ds = len(DATASETS)
n_cols = 2
n_rows = math.ceil(n_ds / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
sns.set_style("whitegrid")  # style

print(f"🚀 Generating convergence curves for {n_ds} datasets...")

for i, ds in enumerate(DATASETS):
    print(f"  > Processing {ds}...", end=" ", flush=True)

    X, y, _ = load_dataset(ds)  # load dataset
    n_feats = X.shape[1]  # number of features

    min_samples = np.min(np.bincount(y))  # min per class
    k_folds = 5 if min_samples >= 5 else max(2, min_samples)  # adjust folds

    evaluador = Evaluator(X, y, 2, int(n_feats * 0.75), k_folds=k_folds, alpha=0.001)  # create evaluator

    _, _, logbook = run_ga(evaluador, n_feats, PARAMS_GA)  # run GA and get log

    gen = logbook.select("gen")  # generations
    fit_max = logbook.select("max")  # best per generation
    fit_avg = logbook.select("avg")  # average per generation

    ax = axes[i]  # current axis
    ax.plot(gen, fit_max, color='#1f77b4', linewidth=2, label='Best Fitness')
    ax.plot(gen, fit_avg, color='#ff7f0e', linestyle='--', linewidth=2, label='Avg. Fitness')

    ax.fill_between(gen, fit_avg, fit_max, color='#1f77b4', alpha=0.1)  # shade difference

    ax.set_title(f'{ds.UPPER()}', fontsize=12, fontweight='bold')  # title
    ax.set_xlabel('Generation')  # X label
    if i % n_cols == 0:
        ax.set_ylabel('Fitness Value')  # Y label in first column

    ax.legend(loc='lower right', fontsize=9)  # legend
    print("✅")

# remove extra axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # adjust layout
nombre_salida = f"{CARPETA_IMG}/convergence_combined.png"  # output path
plt.savefig(nombre_salida, dpi=300)  # save figure
print(f"\n🏁 Plot saved: {nombre_salida}")
plt.show()  # show
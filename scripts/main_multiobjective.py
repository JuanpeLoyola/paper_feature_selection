import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os

from data_loader import load_dataset
from evaluator import Evaluator
from algorithms_mo import run_nsga2

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']

PARAMS_NSGA2 = {'pop_size': 300, 'n_gen': 150, 'p_cruce': 0.6, 'p_mutacion': 0.4}

resultados_pareto = []

print("🚀 STARTING MULTI-OBJECTIVE EXPERIMENT (NSGA-II)...")
print(f"Objectives: Maximize Precision vs Maximize Recall\n")

CARPETA_IMG = "images"
os.makedirs(CARPETA_IMG, exist_ok=True)

for ds in DATASETS:
    try:
        X, y, feat_names = load_dataset(ds)  # download and preprocess
    except Exception as e:
        print(f"❌ Error loading {ds}: {e}")
        continue

    n_feats = X.shape[1]  # number of features

    min_samples_clase = np.min(np.bincount(y))  # minimum per class
    k_folds_dinamico = min(5, min_samples_clase)  # adjust folds
    if k_folds_dinamico < 2:
        k_folds_dinamico = 2  # safety

    print(f"📂 Dataset: {ds} | CV: {k_folds_dinamico}-Folds... ", end="", flush=True)

    k_min = 2  # minimum features
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats  # maximum features

    evaluador = Evaluator(X, y, k_min, k_max, k_folds=k_folds_dinamico)  # create evaluator
    pareto_front, log = run_nsga2(evaluador, n_feats, PARAMS_NSGA2)  # run NSGA-II

    print(f"✅ Pareto Front: {len(pareto_front)} solutions.")

    for i, ind in enumerate(pareto_front):
        prec, recall = ind.fitness.values  # extract objectives
        indices = np.where(np.array(ind) == 1)[0]  # selected indices
        nombres = [feat_names[idx] for idx in indices]  # feature names

        resultados_pareto.append({
            'Dataset': ds,
            'Solution_ID': i,
            'Precision': prec,
            'Recall': recall,
            'N_Features': sum(ind),
            'Feature_Names': str(nombres),
        })

    precisions = [ind.fitness.values[0] for ind in pareto_front]  # precision list
    recalls = [ind.fitness.values[1] for ind in pareto_front]  # recall list

    plt.figure(figsize=(10, 6))  # create figure
    sns.set_style("whitegrid")  # style

    sns.scatterplot(x=precisions, y=recalls, s=100, color='royalblue', edgecolor='k', alpha=0.8)  # points

    plt.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.3, label="Perfect balance")  # ideal line

    #plt.title(f'Pareto Front - {ds.upper()}\n(Precision-Recall Trade-off)', fontsize=14)  # title
    plt.xlabel('Precision', fontsize=12)  # X label
    plt.ylabel('Recall', fontsize=12)  # Y label
    plt.xlim(0.4, 1.05)  # X limits
    plt.ylim(0.4, 1.05)  # Y limits
    plt.grid(True, linestyle='--', alpha=0.7)  # grid

    nombre_imagen = f"{CARPETA_IMG}/pareto_{ds}.png"  # image path
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')  # save
    plt.close()  # close figure

df = pd.DataFrame(resultados_pareto)  # final DataFrame
df.to_csv("csv/multiobjective_results.csv", index=False)  # save CSV
print("\n" + "=" * 60)
print("🏁 Experiment finished.")
print("1. Data saved at: 'csv/multiobjective_results.csv'")
print("2. Images generated: pareto_*.png (Use them in your Chapter 5)")
print("=" * 60)
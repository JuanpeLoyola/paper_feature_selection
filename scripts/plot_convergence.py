import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga
import os

PARAMS_GA = {'pop_size': 110, 'n_gen': 20, 'p_cruce': 0.88, 'p_mutacion': 0.12, 'tam_torneo': 4}  # parámetros GA

DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']  # datasets

CARPETA_IMG = "images"
os.makedirs(CARPETA_IMG, exist_ok=True)  # crear carpeta si no existe

# preparar figura en grid
n_ds = len(DATASETS)
n_cols = 2
n_rows = math.ceil(n_ds / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
sns.set_style("whitegrid")  # estilo

print(f"🚀 Generando curvas de convergencia para {n_ds} datasets...")

for i, ds in enumerate(DATASETS):
    print(f"  > Procesando {ds}...", end=" ", flush=True)

    X, y, _ = cargar_dataset(ds)  # cargar dataset
    n_feats = X.shape[1]  # número de features

    min_samples = np.min(np.bincount(y))  # min por clase
    k_folds = 5 if min_samples >= 5 else max(2, min_samples)  # ajustar folds

    evaluador = Evaluador(X, y, 2, int(n_feats * 0.75), k_folds=k_folds, alpha=0.001)  # crear evaluador

    _, _, logbook = run_ga(evaluador, n_feats, PARAMS_GA)  # ejecutar GA y obtener log

    gen = logbook.select("gen")  # generaciones
    fit_max = logbook.select("max")  # mejor por gen
    fit_avg = logbook.select("avg")  # media por gen

    ax = axes[i]  # eje actual
    ax.plot(gen, fit_max, color='#1f77b4', linewidth=2, label='Best Fitness')
    ax.plot(gen, fit_avg, color='#ff7f0e', linestyle='--', linewidth=2, label='Avg. Fitness')

    ax.fill_between(gen, fit_avg, fit_max, color='#1f77b4', alpha=0.1)  # sombrear diferencia

    ax.set_title(f'{ds.upper()}', fontsize=12, fontweight='bold')  # título
    ax.set_xlabel('Generation')  # etiqueta X
    if i % n_cols == 0:
        ax.set_ylabel('Fitness Value')  # etiqueta Y en primera columna

    ax.legend(loc='lower right', fontsize=9)  # leyenda
    print("✅")

# eliminar ejes sobrantes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # ajustar layout
nombre_salida = f"{CARPETA_IMG}/convergence_combined.png"  # ruta salida
plt.savefig(nombre_salida, dpi=300)  # guardar figura
print(f"\n🏁 Gráfico guardado: {nombre_salida}")
plt.show()  # mostrar
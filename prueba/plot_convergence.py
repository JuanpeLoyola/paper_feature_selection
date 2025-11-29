import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga

# --- CONFIGURACIÓN (Usa tus mejores parámetros de Optuna) ---
PARAMS_GA = {
    'pop_size': 110,    
    'n_gen': 20,        
    'p_cruce': 0.88,     
    'p_mutacion': 0.12,  
    'tam_torneo': 4     
}

DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']

# 1. Configurar la figura compuesta (Grid)
n_ds = len(DATASETS)
n_cols = 2
n_rows = math.ceil(n_ds / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()
sns.set_style("whitegrid")

print(f"🚀 Generando curvas de convergencia para {n_ds} datasets...")

for i, ds in enumerate(DATASETS):
    print(f"  > Procesando {ds}...", end=" ", flush=True)
    
    # Cargar datos
    X, y, _ = cargar_dataset(ds)
    n_feats = X.shape[1]
    
    # Ajuste dinámico de K-Folds (igual que en main)
    min_samples = np.min(np.bincount(y))
    k_folds = 5 if min_samples >= 5 else max(2, min_samples)
    
    # Evaluador con penalización
    evaluador = Evaluador(X, y, 2, int(n_feats*0.75), k_folds=k_folds, alpha=0.001)
    
    # Ejecutar GA y capturar el LOG
    _, _, logbook = run_ga(evaluador, n_feats, PARAMS_GA)
    
    # Extraer datos
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")
    
    # Dibujar en el sub-gráfico correspondiente
    ax = axes[i]
    ax.plot(gen, fit_max, color='#1f77b4', linewidth=2, label='Best Fitness') # Azul
    ax.plot(gen, fit_avg, color='#ff7f0e', linestyle='--', linewidth=2, label='Avg. Fitness') # Naranja
    
    # Relleno entre curvas (muestra la diversidad)
    ax.fill_between(gen, fit_avg, fit_max, color='#1f77b4', alpha=0.1)
    
    ax.set_title(f'{ds.upper()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    if i % n_cols == 0:
        ax.set_ylabel('Fitness Value')
    
    ax.legend(loc='lower right', fontsize=9)
    print("✅")

# Borrar ejes vacíos si sobran
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
nombre_salida = "convergence_combined.png"
plt.savefig(nombre_salida, dpi=300)
print(f"\n🏁 Gráfico guardado: {nombre_salida}")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os

from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms_mo import run_nsga2

# --- SILENCIAR WARNINGS ---
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIGURACIÓN
DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']

# Parámetros para NSGA-II
PARAMS_NSGA2 = {
    'pop_size': 100,    
    'n_gen': 50,        
    'p_cruce': 0.8,
    'p_mutacion': 0.2
}

resultados_pareto = []

print("🚀 INICIANDO EXPERIMENTO MULTIOBJETIVO (NSGA-II)...")
print(f"Objetivos: Maximizar Precisión vs Maximizar Recall\n")

CARPETA_IMG = "imagenes"
os.makedirs(CARPETA_IMG, exist_ok=True)

for ds in DATASETS:
    # 1. Cargar
    try:
        X, y, feat_names = cargar_dataset(ds)
    except Exception as e:
        print(f"❌ Error cargando {ds}: {e}")
        continue

    n_feats = X.shape[1]
    
    # 2. Ajuste Dinámico de K-Folds (Para datasets pequeños como Lymphography)
    min_samples_clase = np.min(np.bincount(y))
    k_folds_dinamico = min(5, min_samples_clase)
    if k_folds_dinamico < 2: k_folds_dinamico = 2
    
    print(f"📂 Dataset: {ds} | CV: {k_folds_dinamico}-Folds... ", end="", flush=True)
    
    # 3. Restricciones
    k_min = 2
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats
    
    # 4. Instanciar Evaluador y Correr
    evaluador = Evaluador(X, y, k_min, k_max, k_folds=k_folds_dinamico)
    pareto_front, log = run_nsga2(evaluador, n_feats, PARAMS_NSGA2)
    
    print(f"✅ Frente de Pareto: {len(pareto_front)} soluciones.")
    
    # 5. Guardar CADA solución del frente
    for i, ind in enumerate(pareto_front):
        prec, recall = ind.fitness.values
        # Convertir índices a nombres de features
        indices = np.where(np.array(ind)==1)[0]
        nombres = [feat_names[idx] for idx in indices]
        
        resultados_pareto.append({
            'Dataset': ds,
            'Solucion_ID': i,
            'Precision': prec,
            'Recall': recall,
            'N_Features': sum(ind),
            'Feature_Names': str(nombres) # Guardamos nombres para que sea legible
        })
        
    # --- GRÁFICO AUTOMÁTICO POR DATASET ---
    precisions = [ind.fitness.values[0] for ind in pareto_front]
    recalls = [ind.fitness.values[1] for ind in pareto_front]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plotear puntos
    sns.scatterplot(x=precisions, y=recalls, s=100, color='royalblue', edgecolor='k', alpha=0.8)
    
    # Línea de "ideal" (opcional)
    plt.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.3, label="Equilibrio perfecto")
    
    plt.title(f'Pareto Front - {ds.upper()}\n(Precision-Recall Trade-off)', fontsize=14)
    plt.xlabel('Precision', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.xlim(0.4, 1.05) # Ajustar ejes para ver mejor la zona alta
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar imagen para el paper
    nombre_imagen = f"{CARPETA_IMG}/pareto_{ds}.png" # <--- CAMBIO AQUÍ
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')
    plt.close()

# Guardar CSV final
df = pd.DataFrame(resultados_pareto)
df.to_csv("resultados_multiobjetivo.csv", index=False)
print("\n" + "="*60)
print("🏁 Experimento terminado.")
print("1. Datos guardados en: 'resultados_multiobjetivo.csv'")
print("2. Imágenes generadas: pareto_*.png (Úsalas en tu Capítulo 5)")
print("="*60)
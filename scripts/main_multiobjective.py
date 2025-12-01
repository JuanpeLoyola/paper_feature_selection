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

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']

PARAMS_NSGA2 = {'pop_size': 100, 'n_gen': 50, 'p_cruce': 0.8, 'p_mutacion': 0.2}

resultados_pareto = []

print("🚀 INICIANDO EXPERIMENTO MULTIOBJETIVO (NSGA-II)...")
print(f"Objetivos: Maximizar Precisión vs Maximizar Recall\n")

CARPETA_IMG = "imagenes"
os.makedirs(CARPETA_IMG, exist_ok=True)

for ds in DATASETS:
    try:
        X, y, feat_names = cargar_dataset(ds)  # descargar y preprocess
    except Exception as e:
        print(f"❌ Error cargando {ds}: {e}")
        continue

    n_feats = X.shape[1]  # número de features

    min_samples_clase = np.min(np.bincount(y))  # mínimo por clase
    k_folds_dinamico = min(5, min_samples_clase)  # ajustar folds
    if k_folds_dinamico < 2:
        k_folds_dinamico = 2  # seguridad

    print(f"📂 Dataset: {ds} | CV: {k_folds_dinamico}-Folds... ", end="", flush=True)

    k_min = 2  # mínimo features
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats  # máximo features

    evaluador = Evaluador(X, y, k_min, k_max, k_folds=k_folds_dinamico)  # crear evaluador
    pareto_front, log = run_nsga2(evaluador, n_feats, PARAMS_NSGA2)  # ejecutar NSGA-II

    print(f"✅ Frente de Pareto: {len(pareto_front)} soluciones.")

    for i, ind in enumerate(pareto_front):
        prec, recall = ind.fitness.values  # extraer objetivos
        indices = np.where(np.array(ind) == 1)[0]  # índices seleccionados
        nombres = [feat_names[idx] for idx in indices]  # nombres de features

        resultados_pareto.append({
            'Dataset': ds,
            'Solucion_ID': i,
            'Precision': prec,
            'Recall': recall,
            'N_Features': sum(ind),
            'Feature_Names': str(nombres),
        })

    precisions = [ind.fitness.values[0] for ind in pareto_front]  # lista precisiones
    recalls = [ind.fitness.values[1] for ind in pareto_front]  # lista recalls

    plt.figure(figsize=(10, 6))  # crear figura
    sns.set_style("whitegrid")  # estilo

    sns.scatterplot(x=precisions, y=recalls, s=100, color='royalblue', edgecolor='k', alpha=0.8)  # puntos

    plt.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.3, label="Equilibrio perfecto")  # línea ideal

    plt.title(f'Pareto Front - {ds.upper()}\n(Precision-Recall Trade-off)', fontsize=14)  # título
    plt.xlabel('Precision', fontsize=12)  # etiqueta X
    plt.ylabel('Recall', fontsize=12)  # etiqueta Y
    plt.xlim(0.4, 1.05)  # límites X
    plt.ylim(0.4, 1.05)  # límites Y
    plt.grid(True, linestyle='--', alpha=0.7)  # rejilla

    nombre_imagen = f"{CARPETA_IMG}/pareto_{ds}.png"  # ruta imagen
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')  # guardar
    plt.close()  # cerrar figura

df = pd.DataFrame(resultados_pareto)  # DataFrame final
df.to_csv("resultados_multiobjetivo.csv", index=False)  # guardar CSV
print("\n" + "=" * 60)
print("🏁 Experimento terminado.")
print("1. Datos guardados en: 'resultados_multiobjetivo.csv'")
print("2. Imágenes generadas: pareto_*.png (Úsalas en tu Capítulo 5)")
print("=" * 60)
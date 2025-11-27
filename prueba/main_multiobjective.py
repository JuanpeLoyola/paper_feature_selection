import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms_mo import run_nsga2

# CONFIGURACIÓN
DATASETS = ['breast_cancer', 'wine', 'ionosphere', 'lymphography', 'zoo']
PARAMS_NSGA2 = {
    'pop_size': 100,    # Población un poco más pequeña para ir rápido
    'n_gen': 50,        # Más generaciones para converger al frente
    'p_cruce': 0.8,
    'p_mutacion': 0.2
}

resultados_pareto = []

print("🚀 INICIANDO EXPERIMENTO MULTIOBJETIVO (NSGA-II)...")

for ds in DATASETS:
    print(f"\n📂 Procesando Dataset: {ds}...")
    
    # 1. Cargar
    X, y, feat_names = cargar_dataset(ds)
    n_feats = X.shape[1]
    
    # 2. Restricciones
    k_min = 2
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats
    
    # 3. Evaluar y Correr
    evaluador = Evaluador(X, y, k_min, k_max)
    pareto_front, log = run_nsga2(evaluador, n_feats, PARAMS_NSGA2)
    
    print(f"   > Frente de Pareto encontrado: {len(pareto_front)} soluciones.")
    
    # 4. Guardar CADA solución del frente
    for i, ind in enumerate(pareto_front):
        prec, recall = ind.fitness.values
        resultados_pareto.append({
            'Dataset': ds,
            'Solucion_ID': i,
            'Precision': prec,
            'Recall': recall,
            'N_Features': sum(ind),
            'Indices': str(np.where(np.array(ind)==1)[0].tolist())
        })
        
    # --- GRÁFICO AUTOMÁTICO POR DATASET ---
    # Sacamos los datos solo de este dataset para plotear ahora mismo
    precisions = [ind.fitness.values[0] for ind in pareto_front]
    recalls = [ind.fitness.values[1] for ind in pareto_front]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=precisions, y=recalls, s=100, color='royalblue', edgecolor='k')
    plt.title(f'Frente de Pareto - {ds}\n(Trade-off Precisión vs Recall)')
    plt.xlabel('Precisión')
    plt.ylabel('Recall')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Guardar imagen
    plt.savefig(f"pareto_{ds}.png")
    plt.close() # Cerrar para no acumular memoria

# Guardar CSV final
df = pd.DataFrame(resultados_pareto)
df.to_csv("resultados_multiobjetivo.csv", index=False)
print("\n✅ Experimento terminado. Datos en 'resultados_multiobjetivo.csv' y gráficos generados.")
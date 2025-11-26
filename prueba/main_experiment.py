import pandas as pd
import time
from data_loader import cargar_dataset
from evaluator import Evaluator
from algorithms import run_ga, run_sa, run_tabu, run_pso

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
DATASETS_A_PROBAR = ['zoo', 'wine', 'ionosphere'] # Añade los que quieras
N_RUNS = 10  # Número de repeticiones para validez estadística (mínimo 10, ideal 30)

# Diccionario de algoritmos
ALGORITMOS = {
    'GA': run_ga,
    'SA': run_sa,
    'Tabu': run_tabu,
    'PSO': run_pso
}

resultados = []

print("🚀 INICIANDO EXPERIMENTO DE FEATURE SELECTION...")
print(f"Datasets: {DATASETS_A_PROBAR}")
print(f"Algoritmos: {list(ALGORITMOS.keys())}")
print(f"Repeticiones por caso: {N_RUNS}\n")

for dataset_name in DATASETS_A_PROBAR:
    # 1. Cargar Datos (una sola vez por dataset)
    try:
        X, y, feat_names = cargar_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Error cargando {dataset_name}: {e}")
        continue
        
    n_feats = X.shape[1]
    
    # Definir restricciones dinámicas
    k_min = 2
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats
    
    # Instanciar Evaluador
    evaluador = Evaluator(X, y, k_min, k_max)
    
    print(f"\n📂 Dataset: {dataset_name} (Features: {n_feats})")
    
    for algo_name, algo_func in ALGORITMOS.items():
        print(f"  🔹 Ejecutando {algo_name}...", end="", flush=True)
        
        for run_id in range(N_RUNS):
            start_time = time.time()
            
            # EJECUCIÓN DEL ALGORITMO
            # Nota: Asegúrate de ajustar los hiperparámetros dentro de algorithms.py
            # para que el coste computacional sea similar (mismo número de evaluaciones aprox)
            best_sol, best_fit = algo_func(evaluador, n_feats)
            
            elapsed_time = time.time() - start_time
            n_selected = sum(best_sol)
            
            # Guardar Resultado
            resultados.append({
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Run_ID': run_id + 1,
                'Best_Precision': best_fit,
                'N_Features_Selected': n_selected,
                'Time_Seconds': elapsed_time,
                'Selected_Features_Indices': str(best_sol) # Guardar como string
            })
            print(".", end="", flush=True)
            
        print(" ✅")

# Guardar a Excel/CSV
df_res = pd.DataFrame(resultados)
archivo_salida = "resultados_feature_selection.csv"
df_res.to_csv(archivo_salida, index=False)

print("\n" + "="*50)
print(f"🏁 Experimento finalizado. Resultados guardados en: {archivo_salida}")
print("="*50)
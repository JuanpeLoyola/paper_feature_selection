import pandas as pd
import time
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga, run_sa, run_tabu, run_pso, run_gwo

# Configuración de hiperparámetros por algoritmo
CONFIGURACION = {
    'GA': {
        'pop_size': 200,    
        'n_gen': 3,         
        'p_cruce': 0.8,     
        'p_mutacion': 0.2,  
        'tam_torneo': 4     
    },
    'SA': {
        'max_iter': 600,
        'alpha': 0.95,
        'temp_init': 1.0
    },
    'Tabu': {
        'max_iter': 60,
        'tabu_size': 10,
        'n_neighbors': 10
    },
    'PSO': {
        'n_particles': 30,  
        'max_iter': 20,
        'w': 0.7, 'c1': 1.5, 'c2': 1.5
    },
    'GWO': {
        'pop_size': 30,
        'max_iter': 20
    }
}

# Configuración del experimento
DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']
N_EJECUCIONES = 10
ALGORITMOS = {
    'GA': run_ga,
    'SA': run_sa,
    'Tabu': run_tabu,
    'PSO': run_pso,
    'GWO': run_gwo 
}

# Ejecución principal
resultados = []
print(f"🚀 Iniciando experimento: {len(DATASETS)} datasets, {len(ALGORITMOS)} algoritmos, {N_EJECUCIONES} ejecuciones\n")

for ds in DATASETS:
    # Cargar dataset
    try:
        X, y, feat_names = cargar_dataset(ds)
    except Exception as e:
        print(f"❌ Error cargando {ds}: {e}")
        continue
    
    # Configurar restricciones de features
    n_feats = X.shape[1]
    k_min = 2
    k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats
    evaluador = Evaluador(X, y, k_min, k_max)
    
    print(f"\n📂 Dataset: {ds} (Features: {n_feats})")
    print("-" * 40)
    
    # Ejecutar cada algoritmo
    for nombre_algo, funcion_algo in ALGORITMOS.items():
        mis_params = CONFIGURACION[nombre_algo]
        print(f"  🔹 {nombre_algo}...", end=" ", flush=True)
        
        # Múltiples ejecuciones para cada algoritmo
        for run_id in range(N_EJECUCIONES):
            start_time = time.time()
            best_sol, best_fit = funcion_algo(evaluador, n_feats, mis_params)
            elapsed = time.time() - start_time
            
            # Calcular número de features seleccionadas
            n_selected = sum(best_sol) if isinstance(best_sol, list) else sum(best_sol > 0.5)
            
            # Almacenar resultados
            resultados.append({
                'Dataset': ds,
                'Algorithm': nombre_algo,
                'Run_ID': run_id + 1,
                'Best_Precision': best_fit,
                'N_Features': n_selected,
                'Time_s': elapsed
            })
            print(".", end="", flush=True)
            
        print(" ✅")

# Guardar resultados en CSV
df_res = pd.DataFrame(resultados)
archivo_salida = "resultados_comparativa_final.csv"
df_res.to_csv(archivo_salida, index=False)

print("\n" + "="*60)
print(f"🏁 Resultados guardados en: {archivo_salida}")
print("="*60)
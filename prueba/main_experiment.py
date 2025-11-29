import pandas as pd
import numpy as np
import time
import sys
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga, run_sa, run_tabu, run_pso, run_gwo
import warnings # <--- NUEVO
from sklearn.exceptions import UndefinedMetricWarning # <--- NUEVO

# --- 1. SILENCIAR WARNINGS MOLESTOS ---
# Ignoramos warnings de métricas (divisiones por cero en modelos malos iniciales)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Para el warning de clases pequeñas


# Configuración de hiperparámetros por algoritmo
# NOTA: Los nombres de los algoritmos deben coincidir con las claves aquí
CONFIGURACION = {
    'GA': {
        'pop_size': 110,    
        'n_gen': 20,         
        'p_cruce': 0.88,     
        'p_mutacion': 0.12,  
        'tam_torneo': 4     
    },
    'SA': {
        'max_iter': 2500,
        'alpha': 0.95,
        'temp_init': 1.0
    },
    'Tabu': {
        'max_iter': 250,
        'tabu_size': 10,
        'n_neighbors': 10
    },
    'PSO': {
        'n_particles': 50,  
        'max_iter': 50,
        'w': 0.7, 'c1': 1.5, 'c2': 1.5
    },
    'GWO': {
        'pop_size': 50,
        'max_iter': 50
    }
}

# Configuración del experimento
DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']
N_EJECUCIONES = 10

# Mapa de nombres a funciones
ALGORITMOS = {
    'GA': run_ga,
    'SA': run_sa,
    'Tabu': run_tabu,
    'PSO': run_pso,
    'GWO': run_gwo 
}

resultados = []
archivo_salida = "resultados_comparativa_final.csv"

print(f"🚀 Iniciando experimento: {len(DATASETS)} datasets, {len(ALGORITMOS)} algoritmos, {N_EJECUCIONES} ejecuciones\n")

ALPHA = 0.001  # Coeficiente de penalización por número de features

try:
    for ds in DATASETS:
        # Cargar dataset
        try:
            X, y, feat_names = cargar_dataset(ds)
        except Exception as e:
            print(f"❌ Error cargando {ds}: {e}")
            continue
        
        # Configurar restricciones de features
        n_feats = X.shape[1]

        # --- 2. AJUSTE DINÁMICO DE K-FOLDS ---
        # Si una clase tiene muy pocos ejemplos (ej: 2), no podemos hacer 5 splits.
        # Ajustamos k_folds al mínimo número de muestras por clase (pero nunca menos de 2)
        min_samples_clase = np.min(np.bincount(y))
        k_folds_dinamico = min(5, min_samples_clase)
        if k_folds_dinamico < 2: k_folds_dinamico = 2 # Seguridad
        
        if k_folds_dinamico < 5:
            print(f"⚠️  Dataset '{ds}' tiene clases pequeñas. Reduciendo CV a {k_folds_dinamico}-Folds.")
        # ---------------------------------------

        # Configurar restricciones de features


        k_min = 2
        k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats
        evaluador = Evaluador(X, y, k_min, k_max, k_folds=k_folds_dinamico, alpha=ALPHA)
        
        print(f"\n📂 Dataset: {ds} (Features: {n_feats})")
        print("-" * 40)
        
        # Ejecutar cada algoritmo
        for nombre_algo, funcion_algo in ALGORITMOS.items():
            mis_params = CONFIGURACION[nombre_algo]
            print(f"  🔹 {nombre_algo}...", end=" ", flush=True)
            
            # Múltiples ejecuciones para cada algoritmo
            for run_id in range(N_EJECUCIONES):
                start_time = time.time()
                
                # AQUI OCURRE LA LLAMADA: Pasamos el evaluador, num features y el DICCIONARIO de parametros
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

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupción detectada. Guardando resultados parciales...")

finally:
    # Guardar resultados en CSV pase lo que pase
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(archivo_salida, index=False)
        print("\n" + "="*60)
        print(f"🏁 Resultados guardados en: {archivo_salida}")
        print("="*60)
    else:
        print("\n❌ No se generaron resultados.")
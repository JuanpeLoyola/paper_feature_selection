import pandas as pd 
import numpy as np  
import time  
import sys  
from data_loader import cargar_dataset 
from evaluator import Evaluador  
from algorithms import run_ga, run_sa, run_tabu, run_pso, run_gwo  
import warnings 
from sklearn.exceptions import UndefinedMetricWarning  
import random  

# ignorar warnings molestos de métricas y usuarios
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# hiperparámetros por algoritmo
CONFIGURACION = {
    'GA': {'pop_size': 110, 'n_gen': 20, 'p_cruce': 0.88, 'p_mutacion': 0.12, 'tam_torneo': 4},
    'SA': {'max_iter': 2500, 'alpha': 0.95, 'temp_init': 1.0},
    'Tabu': {'max_iter': 250, 'tabu_size': 10, 'n_neighbors': 10},
    'PSO': {'n_particles': 50, 'max_iter': 50, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
    'GWO': {'pop_size': 50, 'max_iter': 50},
}

# experiment configuration
DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']  # datasets a evaluar
N_EJECUCIONES = 10  # repeticiones por algoritmo

# mapa de nombre -> función
ALGORITMOS = {'GA': run_ga, 'SA': run_sa, 'Tabu': run_tabu, 'PSO': run_pso, 'GWO': run_gwo}

resultados = []  # acumulador de resultados
archivo_salida = "resultados_comparativa_final.csv"  # CSV de salida

print(f"🚀 Iniciando experimento: {len(DATASETS)} datasets, {len(ALGORITMOS)} algoritmos, {N_EJECUCIONES} ejecuciones\n")

ALPHA = 0.001  # penalización por número de features

try:
    for ds in DATASETS:  # iterar datasets
        try:
            X, y, feat_names = cargar_dataset(ds)  # cargar datos
        except Exception as e:
            print(f"❌ Error cargando {ds}: {e}")  # si falla, saltar
            continue

        n_feats = X.shape[1]  # número total de features

        # ajuste dinámico de folds según la clase con menos muestras
        min_samples_clase = np.min(np.bincount(y))  # mínimo por clase
        k_folds_dinamico = min(5, min_samples_clase)  # usar hasta 5 folds
        if k_folds_dinamico < 2:
            k_folds_dinamico = 2  # seguridad
        if k_folds_dinamico < 5:
            print(f"⚠️  Dataset '{ds}' tiene clases pequeñas. Reduciendo CV a {k_folds_dinamico}-Folds.")

        k_min = 2  # mínimo features seleccionables
        k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats  # máximo permitido
        evaluador = Evaluador(X, y, k_min, k_max, k_folds=k_folds_dinamico, alpha=ALPHA)  # crear evaluador

        print(f"\n📂 Dataset: {ds} (Features: {n_feats})")
        print("-" * 40)

        for nombre_algo, funcion_algo in ALGORITMOS.items():  # ejecutar cada algoritmo
            mis_params = CONFIGURACION[nombre_algo]  # parámetros para el algoritmo
            print(f"  🔹 {nombre_algo}...", end=" ", flush=True)

            for run_id in range(N_EJECUCIONES):  # repeticiones
                semilla_actual = 42 + run_id  # semilla reproducible
                random.seed(semilla_actual)  # semilla global
                np.random.seed(semilla_actual)  # semilla numpy

                start_time = time.time()  # tiempo inicio

                best_sol, best_fit = funcion_algo(evaluador, n_feats, mis_params)  # ejecutar algoritmo

                elapsed = time.time() - start_time  # tiempo transcurrido

                n_selected = sum(best_sol) if isinstance(best_sol, list) else sum(best_sol > 0.5)  # contar features

                resultados.append({  # guardar resultado
                    'Dataset': ds,
                    'Algorithm': nombre_algo,
                    'Run_ID': run_id + 1,
                    'Best_Precision': best_fit,
                    'N_Features': n_selected,
                    'Time_s': elapsed,
                })
                print(".", end="", flush=True)

            print(" ✅")

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupción detectada. Guardando resultados parciales...")  # manejar Ctrl+C

finally:
    if resultados:  # si hay resultados, guardar
        df_res = pd.DataFrame(resultados)  # DataFrame
        df_res.to_csv(archivo_salida, index=False)  # guardar CSV
        print("\n" + "=" * 60)
        print(f"🏁 Resultados guardados en: {archivo_salida}")
        print("=" * 60)
    else:
        print("\n❌ No se generaron resultados.")  # si no hay resultados
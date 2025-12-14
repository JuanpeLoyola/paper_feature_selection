import pandas as pd 
import numpy as np  
import time  
import sys  
from data_loader import load_dataset 
from evaluator import Evaluator  
from algorithms import run_ga, run_sa, run_tabu, run_pso, run_gwo  
import warnings 
from sklearn.exceptions import UndefinedMetricWarning  
import random  

# ignore annoying metric and user warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# hyperparameters per algorithm
CONFIGURACION = {
    'GA': {'pop_size': 110, 'n_gen': 20, 'p_cruce': 0.88, 'p_mutacion': 0.12, 'tam_torneo': 4},
    'SA': {'max_iter': 2500, 'alpha': 0.95, 'temp_init': 1.0},
    'Tabu': {'max_iter': 250, 'tabu_size': 10, 'n_neighbors': 10},
    'PSO': {'n_particles': 50, 'max_iter': 50, 'w': 0.7, 'c1': 1.5, 'c2': 1.5},
    'GWO': {'pop_size': 50, 'max_iter': 50},
}

# experiment configuration
DATASETS = ['zoo', 'wine', 'lymphography', 'ionosphere', 'breast_cancer']  # datasets to evaluate
N_EJECUCIONES = 30  # repetitions per algorithm

# name -> function mapping
ALGORITMOS = {'GA': run_ga, 'SA': run_sa, 'Tabu': run_tabu, 'PSO': run_pso, 'GWO': run_gwo}

resultados = []  # results accumulator
archivo_salida = "csv/final_comparison_results.csv"  # output CSV

print(f"🚀 Starting experiment: {len(DATASETS)} datasets, {len(ALGORITMOS)} algorithms, {N_EJECUCIONES} runs\n")

ALPHA = 0.001  # penalty for number of features

try:
    for ds in DATASETS:  # iterate datasets
        try:
            X, y, feat_names = load_dataset(ds)  # load data
        except Exception as e:
            print(f"❌ Error loading {ds}: {e}")  # if fails, skip
            continue

        n_feats = X.shape[1]  # total number of features

        # dynamic fold adjustment based on smallest class
        min_samples_clase = np.min(np.bincount(y))  # minimum per class
        k_folds_dinamico = min(5, min_samples_clase)  # use up to 5 folds
        if k_folds_dinamico < 2:
            k_folds_dinamico = 2  # safety
        if k_folds_dinamico < 5:
            print(f"⚠️  Dataset '{ds}' has small classes. Reducing CV to {k_folds_dinamico}-Folds.")

        k_min = 2  # minimum selectable features
        k_max = int(n_feats * 0.75) if n_feats > 5 else n_feats  # maximum allowed
        evaluador = Evaluator(X, y, k_min, k_max, k_folds=k_folds_dinamico, alpha=ALPHA)  # create evaluator

        print(f"\n📂 Dataset: {ds} (Features: {n_feats})")
        print("-" * 40)

        for nombre_algo, funcion_algo in ALGORITMOS.items():  # run each algorithm
            mis_params = CONFIGURACION[nombre_algo]  # parameters for algorithm
            print(f"  🔹 {nombre_algo}...", end=" ", flush=True)

            for run_id in range(N_EJECUCIONES):  # repetitions
                semilla_actual = 42 + run_id  # reproducible seed
                random.seed(semilla_actual)  # global seed
                np.random.seed(semilla_actual)  # numpy seed

                start_time = time.time()  # start time

                resultado = funcion_algo(evaluador, n_feats, mis_params)  # run algorithm
                best_sol, best_fit = resultado[0], resultado[1]  # extract solution and fitness

                elapsed = time.time() - start_time  # elapsed time

                n_selected = sum(best_sol) if isinstance(best_sol, list) else sum(best_sol > 0.5)  # count features

                resultados.append({  # save result
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
    print("\n\n⚠️ Interruption detected. Saving partial results...")  # handle Ctrl+C

finally:
    if resultados:  # if there are results, save
        df_res = pd.DataFrame(resultados)  # DataFrame
        df_res.to_csv(archivo_salida, index=False)  # save CSV
        print("\n" + "=" * 60)
        print(f"🏁 Results saved at: {archivo_salida}")
        print("=" * 60)
    else:
        print("\n❌ No results generated.")  # if no results
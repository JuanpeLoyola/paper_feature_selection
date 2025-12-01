import optuna
import numpy as np
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga

DATASET_TUNING = 'ionosphere'  # dataset para tuning
N_TRIALS = 30  # número de pruebas de Optuna

print(f"🎯 Iniciando Tuning de Hiperparámetros con Optuna en '{DATASET_TUNING}'...")

X, y, _ = cargar_dataset(DATASET_TUNING)  # cargar datos una vez
n_feats = X.shape[1]  # número de features

k_min = 2  # mínimo features incluidos
k_max = int(n_feats * 0.75)  # máximo features permitidos
evaluador = Evaluador(X, y, k_min, k_max, k_folds=5, alpha=0.001)  # evaluador


def objective(trial):
    """Objetivo: Optuna sugiere params -> ejecutar GA -> devolver fitness."""
    params = {
        'pop_size': trial.suggest_int('pop_size', 50, 250, step=30),  # población
        'n_gen': trial.suggest_int('n_gen', 30, 100, step=10),  # generaciones
        'p_cruce': trial.suggest_float('p_cruce', 0.5, 0.9),  # prob. cruce
        'p_mutacion': trial.suggest_float('p_mutacion', 0.05, 0.3),  # prob. mutación
        'tam_torneo': trial.suggest_int('tam_torneo', 3, 5),  # tamaño torneo
    }

    try:
        _, best_fit = run_ga(evaluador, n_feats, params)  # ejecutar GA
        return best_fit  # objetivo a maximizar
    except Exception as e:
        print(f"⚠️ Error durante la ejecución de GA con params {params}: {e}")
        return 0.0  # penalizar fallos


study = optuna.create_study(direction='maximize')  # crear estudio
study.optimize(objective, n_trials=N_TRIALS)  # optimizar

print("\n" + "=" * 60)
print("✅ TUNING COMPLETADO")
print("=" * 60)
print(f"Mejor Fitness conseguido: {study.best_value:.4f}")
print("Mejores Hiperparámetros encontrados:")
for key, value in study.best_params.items():
    print(f"   '{key}': {value},")
import optuna
import numpy as np
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga

DATASET_TUNING = 'ionosphere'  # dataset for tuning
N_TRIALS = 30  # number of Optuna trials

print(f"🎯 Starting Hyperparameter Tuning with Optuna on '{DATASET_TUNING}'...")

X, y, _ = cargar_dataset(DATASET_TUNING)  # load data once
n_feats = X.shape[1]  # number of features

k_min = 2  # minimum features included
k_max = int(n_feats * 0.75)  # maximum features allowed
evaluador = Evaluador(X, y, k_min, k_max, k_folds=5, alpha=0.001)  # evaluator


def objective(trial):
    """Objective: Optuna suggests params -> run GA -> return fitness."""
    params = {
        'pop_size': trial.suggest_int('pop_size', 50, 250, step=30),  # population
        'n_gen': trial.suggest_int('n_gen', 30, 100, step=10),  # generations
        'p_cruce': trial.suggest_float('p_cruce', 0.5, 0.9),  # crossover prob.
        'p_mutacion': trial.suggest_float('p_mutacion', 0.05, 0.3),  # mutation prob.
        'tam_torneo': trial.suggest_int('tam_torneo', 3, 5),  # tournament size
    }

    try:
        _, best_fit = run_ga(evaluador, n_feats, params)  # run GA
        return best_fit  # objective to maximize
    except Exception as e:
        print(f"⚠️ Error during GA execution with params {params}: {e}")
        return 0.0  # penalize failures


study = optuna.create_study(direction='maximize')  # create study
study.optimize(objective, n_trials=N_TRIALS)  # optimize

print("\n" + "=" * 60)
print("✅ TUNING COMPLETE")
print("=" * 60)
print(f"Best Fitness achieved: {study.best_value:.4f}")
print("Best Hyperparameters found:")
for key, value in study.best_params.items():
    print(f"   '{key}': {value},")
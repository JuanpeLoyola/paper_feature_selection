import optuna
import numpy as np
from data_loader import cargar_dataset
from evaluator import Evaluador
from algorithms import run_ga

# 1. Configuración
DATASET_TUNING = 'ionosphere' # Usamos este como "banco de pruebas"
N_TRIALS = 30 # Número de experimentos que hará Optuna

print(f"🎯 Iniciando Tuning de Hiperparámetros con Optuna en '{DATASET_TUNING}'...")

# Cargar datos una sola vez
X, y, _ = cargar_dataset(DATASET_TUNING)
n_feats = X.shape[1]

# Definir restricciones estándar para el tuning
k_min = 2
k_max = int(n_feats * 0.75)
evaluador = Evaluador(X, y, k_min, k_max, k_folds=5, alpha=0.001)

def objective(trial):
    """
    Función objetivo para Optuna.
    Optuna sugiere parámetros -> Corremos GA -> Devolvemos fitness.
    """
    # Definir el espacio de búsqueda (Hiperparámetros a optimizar)
    params = {
        # Rango de población: entre 50 y 150
        'pop_size': trial.suggest_int('pop_size', 50, 250, step=30),
        
        # Generaciones: entre 30 y 100
        'n_gen': trial.suggest_int('n_gen', 30, 100, step=10),
        
        # Probabilidad de cruce: entre 0.5 y 0.9
        'p_cruce': trial.suggest_float('p_cruce', 0.5, 0.9),
        
        # Probabilidad de mutación: entre 0.05 y 0.3
        'p_mutacion': trial.suggest_float('p_mutacion', 0.05, 0.3),
        
        # Tamaño del torneo: 3, 4 o 5
        'tam_torneo': trial.suggest_int('tam_torneo', 3, 5)
    }
    
    # Ejecutar GA con estos parámetros
    # Hacemos 3 repeticiones internas para que la aleatoriedad no engañe a Optuna
    try:
        _, best_fit = run_ga(evaluador, n_feats, params)
        return best_fit
    except Exception as e:
        print(f"⚠️ Error durante la ejecución de GA con params {params}: {e}")
        return 0.0 # Si falla por algo, castigamos con 0
            
# Crear el estudio de optimización
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print("\n" + "="*60)
print("✅ TUNING COMPLETADO")
print("="*60)
print(f"Mejor Fitness conseguido: {study.best_value:.4f}")
print("Mejores Hiperparámetros encontrados:")
for key, value in study.best_params.items():
    print(f"   '{key}': {value},")
"""
Algoritmo Genético (GA) para Selección de Características (Feature Selection)
Objetivo: Maximizar la Precisión (Precision) del clasificador.
Restricciones: Número de features entre K_MIN y K_MAX (Hard Constraints).
Modelo Wrapper: Árbol de Clasificación.
"""
#%%
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Librerías de Scikit-learn para el modelo wrapper y evaluación
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score
from sklearn.datasets import load_breast_cancer 
import seaborn as sns

# --- 1. Definición de Constantes y Parámetros ---

# Constantes del Problema (Se deben ajustar a tu dataset)
# Usamos el dataset de cáncer de mama de sklearn como placeholder
datos = load_breast_cancer()
X = datos.data
y = datos.target
N_FEATURES = X.shape[1] # 30 features
K_MIN = 5               # Restricción Dura: Mínimo de features seleccionados
K_MAX = 15              # Restricción Dura: Máximo de features seleccionados
K_FOLDS = 5             # Parámetro para la validación cruzada

# Hiperparámetros del GA (Basado en la metodología solicitada)
TAM_POBLACION = 200
N_GENERACIONES = 50
P_CRUCE = 0.8         # Probabilidad de Cruce (CXPB)
P_MUTACION = 0.2      # Probabilidad de Mutación (MUTPB)
TAM_TORNEO = 4        # Tamaño del Torneo (Selección)

# Random seed
random.seed(42)

# --- 2. Configuración de DEAP ---

# Definimos la estrategia: Maximizar la función objetivo (Peso 1.0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Creamos el individuo (el cromosoma es una lista de genes binarios)
creator.create("Individuo", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Operador para generar un gen (0 o 1)
toolbox.register("gen_binario", random.randint, 0, 1)

# Generador de individuos: Repite N_FEATURES veces la función gen_binario
toolbox.register("generador_individuo", tools.initRepeat, creator.Individuo, toolbox.gen_binario, n=N_FEATURES)

# Generador de población (Lista de individuos)
toolbox.register("poblacionCreator", tools.initRepeat, list, toolbox.generador_individuo)

# --- 3. Función Objetivo (Fitness) ---

def evaluar_FS(individuo, X_data, y_target, k_folds, k_min, k_max):
    """
    Función Objetivo (Fitness) para el GA Wrapper.
    Maximiza la Precisión (Precision) promedio.
    Aplica la penalización por restricciones duras.
    """
    
    # Obtener el número de features seleccionados
    num_features = sum(individuo)
    
    # 1. Aplicar Restricciones Duras (Hard Constraints)
    # Penalización de Muerte (Death Penalty) si se viola el rango [k_min, k_max]
    if num_features < k_min or num_features > k_max:
        return 0.0, # Fitness = 0.0 (inviable)
        
    # Si no se seleccionó ninguna feature (aunque k_min>=1 lo cubre)
    if num_features == 0:
        return 0.0,

    # 2. Seleccionar el subconjunto de features
    indices_seleccionados = np.where(np.array(individuo) == 1)[0]
    X_subconjunto = X_data[:, indices_seleccionados]

    # 3. Modelo Wrapper: Árbol de Clasificación (con hiperparámetros fijos)
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 4. Evaluación: Validación Cruzada (k-Fold)
    precision_scores = cross_val_score(
        modelo, 
        X_subconjunto, 
        y_target, 
        cv=K_FOLDS, 
        scoring='precision',
        error_score=0
    )

    # Devolver la Precisión promedio como la aptitud
    return np.mean(precision_scores),

# --- 4. Registro de Operadores Genéticos ---

toolbox.register("evaluate", evaluar_FS)
# Cruce: Uniforme (tools.cxUniform) - Mayor diversidad
toolbox.register("mate", tools.cxUniform, indpb=0.5) 
# Mutación: Flip-Bit (tools.mutFlipBit) - Ideal para binario
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N_FEATURES) 
# Selección: Torneo (Tamaño 4)
toolbox.register("select", tools.selTournament, tournsize=TAM_TORNEO)

# --- 5. Función Principal del GA (Esquema $\mu+\lambda$) ---

def main():
    
    # Asegurarse de que la función de evaluación esté ligada a los datos
    toolbox.register("evaluate", evaluar_FS, 
                     X_data=X, y_target=y, 
                     k_folds=K_FOLDS, k_min=K_MIN, k_max=K_MAX) 

    # 1. Creación de la población inicial
    pop = toolbox.poblacionCreator(n=TAM_POBLACION) 
    MU, LAMBDA = len(pop), len(pop)
    
    # 2. Inicialización de estadísticas y Hall of Fame
    salon_de_la_fama = tools.HallOfFame(1) 
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values) 
    estadisticas.register("media", np.mean)
    estadisticas.register("min", np.min)
    estadisticas.register("max", np.max)
    logbook = tools.Logbook()
    
    print("Iniciando optimización del Feature Selection...")

    # 3. Loop principal del GA (Esquema $\mu+\lambda$)
    # Los descendientes ($\lambda$) compiten con los padres ($\mu$) por un puesto en la próxima generación.
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                             MU, LAMBDA, 
                                             P_CRUCE, P_MUTACION, 
                                             N_GENERACIONES, 
                                             stats=estadisticas, 
                                             halloffame=salon_de_la_fama)
    
    # --- Resultados ---
    mejor_individuo = salon_de_la_fama[0]
    mejor_aptitud = mejor_individuo.fitness.values[0]
    
    # Encontrar las features seleccionadas por el mejor individuo
    features_seleccionadas_indices = np.where(np.array(mejor_individuo) == 1)[0]
    nombres_features = datos.feature_names[features_seleccionadas_indices]
    
    print("\n" + "="*60)
    print("✅ Optimización del Feature Selection Completada")
    print("="*60)
    print(f"Mejor Precisión (Aptitud): {mejor_aptitud:.4f}")
    print(f"Número de Features Seleccionadas: {len(features_seleccionadas_indices)}")
    print(f"Features Seleccionadas:")
    for i, nombre in enumerate(nombres_features):
        print(f"  {i+1}. {nombre}")


    # Ploteo de la Evolución
    plot_evolucion(logbook)
    
    return mejor_individuo, logbook

def plot_evolucion(log):
    gen = log.select("gen")
    fit_maxs = log.select("max")
    fit_ave = log.select("media")
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_maxs, color='red', label='Precisión Máxima')
    plt.plot(gen, fit_ave, color='green', linestyle='--', label='Precisión Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Precisión (Aptitud)')
    plt.title('Evolución de la Precisión Máxima y Promedio')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
# %%

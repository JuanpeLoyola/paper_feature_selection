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
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer 
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml, load_wine
# SImpleImputer para manejar valores faltantes
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
# Las dos líneas anteriores son para ignorar warnings molestos de DEAP sobre la creación de clases.

# --- 1. Definición de Constantes y Parámetros ---

# Dataset
# Cambia esta variable para probar otro dataset: 'breast_cancer', 'wine', 'ionosphere', 'lymphography', 'zoo', 'parkinsons'
NOMBRE_DATASET = 'zoo' 

def cargar_dataset(nombre):
    print(f"📥 Cargando dataset: {nombre}...")
    
    datasets_openml = {
        'zoo': 966, 'congress_ew': 31, 'vote': 31, 'breast_ew': 15, 
        'breast_cancer': 1510, 'wine': 187, 'lung': 32, 'm-of-n': 934, 
        'heart_ew': 53, 'spect_ew': 951, 'lymphography': 10, 
        'ionosphere': 59, 'sonar': 40, 'parkinsons': 1488
    }
    
    feature_names = None 

    if nombre in datasets_openml:
        bunch = fetch_openml(data_id=datasets_openml[nombre], as_frame=True, parser='auto')
        X = bunch.data
        y = bunch.target
        
        if hasattr(X, 'columns'):
            feature_names = X.columns.astype(str).tolist()
        else:
            feature_names = bunch.feature_names
            
    elif nombre == 'breast_cancer_sklearn': 
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names.tolist()
    else:
        raise ValueError(f"Dataset '{nombre}' no reconocido.")

    # --- Preprocesamiento Universal ---
    if hasattr(X, 'iloc'): 
        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.astype(str).tolist() 
        X = X.values 
    
    X = np.array(X, dtype=float)
    
    if feature_names is None or len(feature_names) != X.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    if np.isnan(X).any():
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)
        
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, feature_names

# --- CARGA DE DATOS ---
X, y, NOMBRES_FEATURES = cargar_dataset(NOMBRE_DATASET)

# --- CONSTANTES Y RESTRICCIONES ---
N_FEATURES = X.shape[1] 
K_FOLDS = 5             # Validación cruzada (se mantiene fijo)

# Ajuste dinámico de restricciones (Vital para datasets pequeños como Wine)
K_MIN = 2
K_MAX = int(N_FEATURES * 0.75) # Máximo el 75% de las features
if K_MAX < K_MIN: K_MAX = N_FEATURES 
if K_MAX > 25: K_MAX = 25 # (Opcional) Tope superior para no tardar mucho en datasets gigantes

print(f"⚙️ Restricciones aplicadas: Min={K_MIN}, Max={K_MAX} features (de {N_FEATURES} totales).")

# Hiperparámetros del GA
TAM_POBLACION = 200   # Tamaño de la Población
N_GENERACIONES = 3   # Número de Generaciones
P_CRUCE = 0.8         # Probabilidad de Cruce
P_MUTACION = 0.2      # Probabilidad de Mutación
TAM_TORNEO = 4        # Tamaño del Torneo para Selección

# Random seed para reproducibilidad
random.seed(42)

# --- 2. Configuración de DEAP ---

# Definimos la estrategia: Maximizar la función objetivo (Peso 1.0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Creamos el individuo (el cromosoma es una lista de genes binarios que representan la selección de features)
creator.create("Individuo", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox() # Esto es para registrar los operadores genéticos

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

    # 2. Seleccionar el subconjunto de features
    indices_seleccionados = np.where(np.array(individuo) == 1)[0]
    X_subconjunto = X_data[:, indices_seleccionados]

    # 3. Modelo Wrapper: Árbol de Clasificación (con hiperparámetros por defecto)
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 4. Evaluación: Validación Cruzada (k-Fold)
    precision_scores = cross_val_score(
        modelo, 
        X_subconjunto, 
        y_target, 
        cv=K_FOLDS, 
        scoring='precision_weighted', # Esto es para que funcione en clasificación multiclase. Calcula la presición ponderada dándole peso a cada clase según su frecuencia.
        error_score=0
    )

    # Devolver la Precisión promedio como la aptitud
    return np.mean(precision_scores),


# --- 4. Registro de Operadores Genéticos ---

# Selección: Torneo (Tamaño 4)
toolbox.register("select", tools.selTournament, tournsize=TAM_TORNEO)

# Cruce: Uniforme (tools.cxUniform) - Mayor diversidad
toolbox.register("mate", tools.cxUniform, indpb=0.5) 

# Mutación: Flip-Bit (tools.mutFlipBit) - Ideal para binario
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N_FEATURES) # Con esta probabilidad nos aseguramos que en promedio se muta 1 gen por individuo


# --- 5. Función Principal del GA (Esquema MuPlusLambda) ---

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

    # 3. Bucle principal del GA (Esquema MuPlusLambda)

    # Los descendientes (Lambda) compiten con los padres (Mu) por un puesto en la próxima generación.
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
    nombres_seleccionados = np.array(NOMBRES_FEATURES)[features_seleccionadas_indices] # Convertimos a array de numpy temporalmente para poder indexar con la lista de índices
    
    print("\n" + "="*60)
    print("✅ Optimización del Feature Selection Completada")
    print("="*60)
    print(f"Mejor Precisión (Fitness): {mejor_aptitud:.4f}")
    print(f"Número de Features Seleccionadas: {len(features_seleccionadas_indices)}")
    print(f"Features Seleccionadas:")
    for i, nombre in enumerate(nombres_seleccionados):
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
    plt.ylabel('Precisión (Fitness)')
    plt.title('Evolución de la Precisión Máxima y Promedio')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

# %%
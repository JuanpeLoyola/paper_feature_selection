"""
Algoritmo Genético Multiobjetivo (NSGA-II) para Selección de Características (Feature Selection)
Objetivos: Maximizar la Precisión (Precision) y el Recall del clasificador.
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
N_GENERACIONES = 10   # Número de Generaciones
P_CRUCE = 0.8         # Probabilidad de Cruce
P_MUTACION = 0.2      # Probabilidad de Mutación
TAM_TORNEO = 4        # Tamaño del Torneo para Selección

# Random seed para reproducibilidad
random.seed(42)

# --- 2. Configuración de DEAP ---

# Definimos la estrategia: Maximizar AMBOS objetivos (Precisión y Recall) - Pesos (1.0, 1.0)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
# Creamos el individuo (el cromosoma es una lista de genes binarios que representan la selección de features)
creator.create("Individuo", list, fitness=creator.FitnessMulti)

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
    Función Objetivo (Fitness) para el GA Wrapper Multiobjetivo.
    Maximiza la Precisión (Precision) y el Recall promedio.
    Aplica la penalización por restricciones duras.
    """
    
    # Obtener el número de features seleccionados
    num_features = sum(individuo)
    
    # 1. Aplicar Restricciones Duras (Hard Constraints)
    # Penalización de Muerte (Death Penalty) si se viola el rango [k_min, k_max]
    if num_features < k_min or num_features > k_max:
        return 0.0, 0.0 # Fitness = (0.0, 0.0) para ambos objetivos (inviable)

    # 2. Seleccionar el subconjunto de features
    indices_seleccionados = np.where(np.array(individuo) == 1)[0]
    X_subconjunto = X_data[:, indices_seleccionados]

    # 3. Modelo Wrapper: Árbol de Clasificación (con hiperparámetros por defecto)
    modelo = DecisionTreeClassifier(random_state=42)
    
    # 4. Evaluación: Validación Cruzada (k-Fold) para Precisión
    precision_scores = cross_val_score(
        modelo, 
        X_subconjunto, 
        y_target, 
        cv=K_FOLDS, 
        scoring='precision_weighted', # Precisión ponderada para clasificación multiclase
        error_score=0
    )
    
    # 5. Evaluación: Validación Cruzada (k-Fold) para Recall
    recall_scores = cross_val_score(
        modelo, 
        X_subconjunto, 
        y_target, 
        cv=K_FOLDS, 
        scoring='recall_weighted', # Recall ponderado para clasificación multiclase
        error_score=0
    )

    # Devolver la Precisión promedio y el Recall promedio como la tupla de aptitudes
    return np.mean(precision_scores), np.mean(recall_scores)


# --- 4. Registro de Operadores Genéticos ---

# Selección: NSGA-II (específico para optimización multiobjetivo)
toolbox.register("select", tools.selNSGA2)

# Cruce: Uniforme (tools.cxUniform) - Mayor diversidad
toolbox.register("mate", tools.cxUniform, indpb=0.5) 

# Mutación: Flip-Bit (tools.mutFlipBit) - Ideal para binario
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N_FEATURES) # Con esta probabilidad nos aseguramos que en promedio se muta 1 gen por individuo


# --- 5. Función Principal del GA (NSGA-II) ---

def main():
    
    # Asegurarse de que la función de evaluación esté ligada a los datos
    toolbox.register("evaluate", evaluar_FS, 
                     X_data=X, y_target=y, 
                     k_folds=K_FOLDS, k_min=K_MIN, k_max=K_MAX) 

    # 1. Creación de la población inicial
    pop = toolbox.poblacionCreator(n=TAM_POBLACION) 
    
    # 2. Inicialización de estadísticas y Pareto Front
    pareto_front = tools.ParetoFront() # Para almacenar el frente de Pareto
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values) 
    estadisticas.register("media", np.mean, axis=0)
    estadisticas.register("min", np.min, axis=0)
    estadisticas.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    
    print("Iniciando optimización multiobjetivo del Feature Selection con NSGA-II...")

    # 3. Bucle principal del GA (Algoritmo NSGA-II)
    # NSGA-II es específico para optimización multiobjetivo
    # Usamos eaMuPlusLambda con selNSGA2 para implementar NSGA-II
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                             TAM_POBLACION, TAM_POBLACION, 
                                             P_CRUCE, P_MUTACION, 
                                             N_GENERACIONES, 
                                             stats=estadisticas, 
                                             halloffame=pareto_front,
                                             verbose=True)
    
    # --- Resultados ---
    print("\n" + "="*60)
    print("✅ Optimización Multiobjetivo del Feature Selection Completada")
    print("="*60)
    print(f"Tamaño del Frente de Pareto: {len(pareto_front)}")
    print("\nSoluciones del Frente de Pareto:")
    print("-" * 60)
    
    # Mostrar todas las soluciones del frente de Pareto
    for i, individuo in enumerate(pareto_front):
        precision, recall = individuo.fitness.values
        num_features = sum(individuo)
        print(f"Solución {i+1}: Precisión={precision:.4f}, Recall={recall:.4f}, Features={num_features}")
    
    # Seleccionar algunas soluciones representativas para mostrar sus features
    print("\n" + "="*60)
    print("📊 Análisis de Soluciones Representativas")
    print("="*60)
    
    # Solución con mejor precisión
    mejor_precision_idx = np.argmax([ind.fitness.values[0] for ind in pareto_front])
    mejor_precision_ind = pareto_front[mejor_precision_idx]
    
    print(f"\n🎯 Solución con Mejor Precisión:")
    print(f"   Precisión: {mejor_precision_ind.fitness.values[0]:.4f}")
    print(f"   Recall: {mejor_precision_ind.fitness.values[1]:.4f}")
    print(f"   Número de Features: {sum(mejor_precision_ind)}")
    features_idx = np.where(np.array(mejor_precision_ind) == 1)[0]
    nombres_sel = np.array(NOMBRES_FEATURES)[features_idx]
    print(f"   Features seleccionadas:")
    for j, nombre in enumerate(nombres_sel):
        print(f"     {j+1}. {nombre}")
    
    # Solución con mejor recall
    mejor_recall_idx = np.argmax([ind.fitness.values[1] for ind in pareto_front])
    mejor_recall_ind = pareto_front[mejor_recall_idx]
    
    print(f"\n🎯 Solución con Mejor Recall:")
    print(f"   Precisión: {mejor_recall_ind.fitness.values[0]:.4f}")
    print(f"   Recall: {mejor_recall_ind.fitness.values[1]:.4f}")
    print(f"   Número de Features: {sum(mejor_recall_ind)}")
    features_idx = np.where(np.array(mejor_recall_ind) == 1)[0]
    nombres_sel = np.array(NOMBRES_FEATURES)[features_idx]
    print(f"   Features seleccionadas:")
    for j, nombre in enumerate(nombres_sel):
        print(f"     {j+1}. {nombre}")
    
    # Solución balanceada (punto medio del frente de Pareto)
    if len(pareto_front) > 2:
        medio_idx = len(pareto_front) // 2
        medio_ind = pareto_front[medio_idx]
        
        print(f"\n⚖️  Solución Balanceada (punto medio):")
        print(f"   Precisión: {medio_ind.fitness.values[0]:.4f}")
        print(f"   Recall: {medio_ind.fitness.values[1]:.4f}")
        print(f"   Número de Features: {sum(medio_ind)}")
        features_idx = np.where(np.array(medio_ind) == 1)[0]
        nombres_sel = np.array(NOMBRES_FEATURES)[features_idx]
        print(f"   Features seleccionadas:")
        for j, nombre in enumerate(nombres_sel):
            print(f"     {j+1}. {nombre}")

    # Ploteo de resultados
    plot_pareto_front(pareto_front)
    plot_evolucion_multiobjetivo(logbook)
    
    return pareto_front, logbook

def plot_pareto_front(pareto_front):
    """Visualiza el frente de Pareto en el espacio de objetivos (Precisión vs Recall)"""
    precision_vals = [ind.fitness.values[0] for ind in pareto_front]
    recall_vals = [ind.fitness.values[1] for ind in pareto_front]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    plt.scatter(precision_vals, recall_vals, c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    plt.xlabel('Precisión (Precision)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Frente de Pareto: Precisión vs Recall', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Añadir etiquetas a algunos puntos representativos
    if len(pareto_front) > 0:
        # Mejor precisión
        mejor_prec_idx = np.argmax(precision_vals)
        plt.annotate('Mejor Precisión', 
                    xy=(precision_vals[mejor_prec_idx], recall_vals[mejor_prec_idx]),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Mejor recall
        mejor_rec_idx = np.argmax(recall_vals)
        plt.annotate('Mejor Recall', 
                    xy=(precision_vals[mejor_rec_idx], recall_vals[mejor_rec_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()

def plot_evolucion_multiobjetivo(log):
    """Visualiza la evolución de ambos objetivos a lo largo de las generaciones"""
    gen = log.select("gen")
    fit_maxs = np.array(log.select("max"))
    fit_ave = np.array(log.select("media"))
    
    # Extraer precisión y recall
    max_precision = fit_maxs[:, 0]
    max_recall = fit_maxs[:, 1]
    ave_precision = fit_ave[:, 0]
    ave_recall = fit_ave[:, 1]
    
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Evolución de Precisión
    ax1.plot(gen, max_precision, color='red', linewidth=2, label='Precisión Máxima')
    ax1.plot(gen, ave_precision, color='orange', linestyle='--', linewidth=2, label='Precisión Promedio')
    ax1.set_xlabel('Generación', fontsize=12)
    ax1.set_ylabel('Precisión', fontsize=12)
    ax1.set_title('Evolución de la Precisión', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Evolución de Recall
    ax2.plot(gen, max_recall, color='blue', linewidth=2, label='Recall Máximo')
    ax2.plot(gen, ave_recall, color='cyan', linestyle='--', linewidth=2, label='Recall Promedio')
    ax2.set_xlabel('Generación', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Evolución del Recall', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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

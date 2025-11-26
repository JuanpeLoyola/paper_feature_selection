import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Este script se encarga de evaluar soluciones propuestas por los algoritmos

class Evaluador:
    def __init__(self, X, y, k_min, k_max, k_folds=5):
        self.X = X
        self.y = y
        self.k_min = k_min
        self.k_max = k_max
        self.k_folds = k_folds
        # Tu modelo base
        self.modelo = DecisionTreeClassifier(random_state=42)

    def evaluar(self, individuo):
        """
        Función Objetivo unificada.
        Acepta tanto listas de 0/1 (GA, SA, Tabu) como arrays de floats (PSO).
        """
        # Adaptación para PSO (si llega float, lo convertimos a bool)
        if isinstance(individuo[0], float) or isinstance(individuo[0], np.float64): 
            mask = np.array(individuo) > 0.5
        else:
            mask = np.array(individuo) == 1
            
        num_features = sum(mask)
        
        # 1. Aplicar Restricciones Duras (Tu lógica)
        if num_features < self.k_min or num_features > self.k_max:
            return 0.0, # Fitness = 0.0 (inviable)

        # 2. Seleccionar subconjunto
        indices_seleccionados = np.where(mask)[0]
        X_subconjunto = self.X[:, indices_seleccionados]

        # 3. Evaluación: Validación Cruzada
        precision_scores = cross_val_score(
            self.modelo, 
            X_subconjunto, 
            self.y, 
            cv=self.k_folds, 
            scoring='precision_weighted', 
            error_score=0
        )

        return np.mean(precision_scores),
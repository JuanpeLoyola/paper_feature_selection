import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class Evaluator:
    def __init__(self, X, y, k_min, k_max, k_folds=5):
        self.X = X
        self.y = y
        self.k_min = k_min
        self.k_max = k_max
        self.k_folds = k_folds
        # Modelo fijo para todos los algoritmos
        self.clf = DecisionTreeClassifier(random_state=42)

    def evaluate(self, individual):
        """
        Recibe un vector binario (lista o array de 0s y 1s).
        Devuelve (Precision,).
        """
        # Convertir a array booleano
        mask = np.array(individual) > 0.5
        n_features = sum(mask)
        
        # --- Hard Constraints ---
        if n_features < self.k_min or n_features > self.k_max:
            return 0.0, # Penalización total

        # --- Validación Cruzada ---
        X_subset = self.X[:, mask]
        
        # Scoring 'precision_weighted' para multiclase
        scores = cross_val_score(self.clf, X_subset, self.y, 
                                 cv=self.k_folds, scoring='precision_weighted')
        
        return np.mean(scores),
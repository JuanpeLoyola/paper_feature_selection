import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class Evaluador:
    """Evalúa soluciones de selección de features mediante validación cruzada."""
    
    def __init__(self, X, y, k_min, k_max, k_folds=5):
        self.X = X
        self.y = y
        self.k_min = k_min
        self.k_max = k_max
        self.k_folds = k_folds
        self.modelo = DecisionTreeClassifier(random_state=42)

    def evaluar(self, individuo):
        """Calcula fitness de un individuo (precisión media por validación cruzada)."""
        # Conversión de individuo a máscara booleana
        mask = np.array(individuo) > 0.5 if isinstance(individuo[0], (float, np.float64)) else np.array(individuo) == 1
        num_features = sum(mask)
        
        # Restricción: número de features debe estar en [k_min, k_max]
        if num_features < self.k_min or num_features > self.k_max:
            return 0.0,
        
        # Selección de features
        X_subconjunto = self.X[:, mask]
        
        # Evaluación mediante validación cruzada
        precision_scores = cross_val_score(
            self.modelo, 
            X_subconjunto, 
            self.y, 
            cv=self.k_folds, 
            scoring='precision_weighted', 
            error_score=0
        )

        return np.mean(precision_scores),

    def evaluar_multiobjetivo(self, individuo):
            """Para Multi Objective (NSGA-II). Retorna (Precision, Recall)"""
            mask = np.array(individuo) == 1
            num_features = sum(mask)
            
            # Hard Constraints (Tu lógica)
            if num_features < self.k_min or num_features > self.k_max:
                return 0.0, 0.0 # Penalización doble

            indices = np.where(mask)[0]
            X_sub = self.X[:, indices]

            # 1. Precisión
            prec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='precision_weighted', error_score=0)
            # 2. Recall
            rec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='recall_weighted', error_score=0)

            return np.mean(prec_scores), np.mean(rec_scores)
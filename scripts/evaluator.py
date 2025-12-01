import numpy as np  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score  


class Evaluador:
    """Evalúa máscaras de selección de features via CV."""

    def __init__(self, X, y, k_min, k_max, k_folds=5, alpha=0.0):
        self.X = X  # matriz de características
        self.y = y  # vector de etiquetas
        self.k_min = k_min  # mínimo de features permitidos
        self.k_max = k_max  # máximo de features permitidos
        self.k_folds = k_folds  # folds para CV
        self.alpha = alpha  # penalización por número de features
        self.modelo = DecisionTreeClassifier(random_state=42)  # modelo base

    def evaluar(self, individuo):
        """Devolver (precision penalizada,) para un individuo binario."""
        mask = np.array(individuo) > 0.5 if isinstance(individuo[0], (float, np.float64)) else np.array(individuo) == 1  # máscara booleana
        num_features = sum(mask)  # contar features seleccionadas

        if num_features < self.k_min or num_features > self.k_max:  # constraint
            return 0.0,  # penalizar si fuera de rango

        X_subconjunto = self.X[:, mask]  # aplicar máscara

        precision_scores = cross_val_score(
            self.modelo,
            X_subconjunto,
            self.y,
            cv=self.k_folds,
            scoring='precision_weighted',
            error_score=0,
        )  # obtener precisiones por fold

        promedio_precision = np.mean(precision_scores)  # media de precisión

        return promedio_precision - self.alpha * num_features,  # devolver fitness penalizado

    def evaluar_multiobjetivo(self, individuo):
        """Retorna (precision, recall) para NSGA-II."""
        mask = np.array(individuo) == 1  # máscara booleana
        num_features = sum(mask)  # contar

        if num_features < self.k_min or num_features > self.k_max:  # constraint
            return 0.0, 0.0  # penalizar ambos objetivos

        indices = np.where(mask)[0]  # índices seleccionados
        X_sub = self.X[:, indices]  # subconjunto de X

        prec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='precision_weighted', error_score=0)  # precisión
        rec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='recall_weighted', error_score=0)  # recall

        return np.mean(prec_scores), np.mean(rec_scores)  # devolver tupla (prec, rec)
import numpy as np  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score  


class Evaluador:
    """Evaluates feature selection masks via CV."""

    def __init__(self, X, y, k_min, k_max, k_folds=5, alpha=0.0):
        self.X = X  # feature matrix
        self.y = y  # label vector
        self.k_min = k_min  # minimum features allowed
        self.k_max = k_max  # maximum features allowed
        self.k_folds = k_folds  # folds for CV
        self.alpha = alpha  # penalty for number of features
        self.modelo = DecisionTreeClassifier(random_state=42)  # base model

    def evaluar(self, individuo):
        """Return (penalized precision,) for a binary individual."""
        mask = np.array(individuo) > 0.5 if isinstance(individuo[0], (float, np.float64)) else np.array(individuo) == 1  # boolean mask
        num_features = sum(mask)  # count selected features

        if num_features < self.k_min or num_features > self.k_max:  # constraint
            return 0.0,  # penalize if out of range

        X_subconjunto = self.X[:, mask]  # apply mask

        precision_scores = cross_val_score(
            self.modelo,
            X_subconjunto,
            self.y,
            cv=self.k_folds,
            scoring='precision_weighted',
            error_score=0,
        )  # get precision per fold

        promedio_precision = np.mean(precision_scores)  # average precision

        return promedio_precision - self.alpha * num_features,  # return penalized fitness

    def evaluar_multiobjetivo(self, individuo):
        """Return (precision, recall) for NSGA-II."""
        mask = np.array(individuo) == 1  # boolean mask
        num_features = sum(mask)  # count

        if num_features < self.k_min or num_features > self.k_max:  # constraint
            return 0.0, 0.0  # penalize both objectives

        indices = np.where(mask)[0]  # selected indices
        X_sub = self.X[:, indices]  # X subset

        prec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='precision_weighted', error_score=0)  # precision
        rec_scores = cross_val_score(self.modelo, X_sub, self.y, cv=self.k_folds, scoring='recall_weighted', error_score=0)  # recall

        return np.mean(prec_scores), np.mean(rec_scores)  # return tuple (prec, rec)
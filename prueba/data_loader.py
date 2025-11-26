import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def cargar_dataset(nombre):
    print(f"📥 Cargando dataset: {nombre}...")
    
    # IDs de OpenML para datasets comunes en FS
    datasets_openml = {
        'zoo': 966, 'vote': 31, 'breast_cancer': 1510, 
        'wine': 187, 'ionosphere': 59, 'sonar': 40, 'parkinsons': 1488
    }
    
    X, y, feature_names = None, None, None

    if nombre in datasets_openml:
        bunch = fetch_openml(data_id=datasets_openml[nombre], as_frame=True, parser='auto')
        X = bunch.data
        y = bunch.target
        feature_names = bunch.feature_names
    elif nombre == 'breast_sklearn':
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names.tolist()
    elif nombre == 'wine_sklearn':
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
    else:
        raise ValueError(f"Dataset '{nombre}' no reconocido.")

    # Preprocesamiento
    if hasattr(X, 'iloc'):
        X = pd.get_dummies(X, drop_first=True)
        feature_names = X.columns.astype(str).tolist()
        X = X.values
        
    X = np.array(X, dtype=float)
    
    # Imputar nulos
    if np.isnan(X).any():
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)
        
    # Codificar Target
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(X.shape[1])]

    return X, y, np.array(feature_names)
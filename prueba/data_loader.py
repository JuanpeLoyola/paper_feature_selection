import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Este script se encarga de cargar y preprocesar los datasets

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

    return X, y, np.array(feature_names)
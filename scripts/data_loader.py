import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def cargar_dataset(nombre):
    """Carga y preprocesa datasets desde OpenML."""
    print(f"📥 Cargando dataset: {nombre}...")
    
    # Mapeo de nombres a IDs de OpenML
    datasets_openml = {
        'zoo': 966, 'congress_ew': 31, 'vote': 31, 'breast_ew': 15, 
        'breast_cancer': 1510, 'wine': 187, 'lung': 32, 'm-of-n': 934, 
        'heart_ew': 53, 'spect_ew': 951, 'lymphography': 10, 
        'ionosphere': 59, 'sonar': 40, 'parkinsons': 1488
    }
    
    if nombre not in datasets_openml:
        raise ValueError(f"Dataset '{nombre}' no reconocido.")
    
    # Descarga del dataset
    bunch = fetch_openml(data_id=datasets_openml[nombre], as_frame=True, parser='auto')
    X = bunch.data
    y = bunch.target
    
    # Extracción de nombres de features
    feature_names = X.columns.astype(str).tolist() if hasattr(X, 'columns') else bunch.feature_names
    
    # Conversión de variables categóricas a dummy
    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.astype(str).tolist()
    X = X.values
    
    # Conversión a array numérico
    X = np.array(X, dtype=float)
    
    # Imputación de valores faltantes
    if np.isnan(X).any():
        X = SimpleImputer(strategy='mean').fit_transform(X)
    
    # Codificación de etiquetas
    y = LabelEncoder().fit_transform(y)

    return X, y, np.array(feature_names)
import pandas as pd 
import numpy as np  
from sklearn.datasets import fetch_openml  
from sklearn.preprocessing import LabelEncoder  
from sklearn.impute import SimpleImputer  


def cargar_dataset(nombre):
    """Cargar y preprocesar un dataset desde OpenML."""
    print(f"📥 Cargando dataset: {nombre}...")  # indicar inicio

    datasets_openml = {  # mapeo nombre -> id OpenML
        'zoo': 966, 'congress_ew': 31, 'vote': 31, 'breast_ew': 15,
        'breast_cancer': 1510, 'wine': 187, 'lung': 32, 'm-of-n': 934,
        'heart_ew': 53, 'spect_ew': 951, 'lymphography': 10,
        'ionosphere': 59, 'sonar': 40, 'parkinsons': 1488
    }

    if nombre not in datasets_openml:
        raise ValueError(f"Dataset '{nombre}' no reconocido.")  # validar nombre

    bunch = fetch_openml(data_id=datasets_openml[nombre], as_frame=True, parser='auto')  # descargar
    X = bunch.data  # características como DataFrame
    y = bunch.target  # etiquetas

    feature_names = X.columns.astype(str).tolist() if hasattr(X, 'columns') else bunch.feature_names  # nombres

    X = pd.get_dummies(X, drop_first=True)  # one-hot para categóricas
    feature_names = X.columns.astype(str).tolist()  # actualizar nombres
    X = X.values  # convertir a ndarray

    X = np.array(X, dtype=float)  # asegurar tipo float

    if np.isnan(X).any():
        X = SimpleImputer(strategy='mean').fit_transform(X)  # imputar NaNs

    y = LabelEncoder().fit_transform(y)  # codificar etiquetas a enteros

    return X, y, np.array(feature_names)  # devolver X, y y nombres
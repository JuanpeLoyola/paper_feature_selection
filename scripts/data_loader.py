import pandas as pd 
import numpy as np  
from sklearn.datasets import fetch_openml  
from sklearn.preprocessing import LabelEncoder  
from sklearn.impute import SimpleImputer  


def load_dataset(nombre):
    """Load and preprocess a dataset from OpenML."""
    print(f"📥 Loading dataset: {nombre}...")  # indicate start

    datasets_openml = {  # mapping name -> OpenML id
        'zoo': 966, 'congress_ew': 31, 'vote': 31, 'breast_ew': 15,
        'breast_cancer': 1510, 'wine': 187, 'lung': 32, 'm-of-n': 934,
        'heart_ew': 53, 'spect_ew': 951, 'lymphography': 10,
        'ionosphere': 59, 'sonar': 40, 'parkinsons': 1488
    }

    if nombre not in datasets_openml:
        raise ValueError(f"Dataset '{nombre}' not recognized.")  # validate name

    bunch = fetch_openml(data_id=datasets_openml[nombre], as_frame=True, parser='auto')  # download
    X = bunch.data  # features as DataFrame
    y = bunch.target  # labels

    feature_names = X.columns.astype(str).tolist() if hasattr(X, 'columns') else bunch.feature_names  # names

    X = pd.get_dummies(X, drop_first=True)  # one-hot for categorical
    feature_names = X.columns.astype(str).tolist()  # update names
    X = X.values  # convert to ndarray

    X = np.array(X, dtype=float)  # ensure float type

    if np.isnan(X).any():
        X = SimpleImputer(strategy='mean').fit_transform(X)  # impute NaNs

    y = LabelEncoder().fit_transform(y)  # encode labels to integers

    return X, y, np.array(feature_names)  # return X, y and names
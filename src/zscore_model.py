import numpy as np

def zscore_anomaly(X, threshold=3):
    z_scores = np.abs((X - X.mean()) / X.std())
    anomaly = (z_scores > threshold).any(axis=1)
    return anomaly.astype(int)
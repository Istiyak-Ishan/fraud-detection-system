from sklearn.neighbors import LocalOutlierFactor

def train_lof(X):
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        novelty=True   # important for prediction
    )
    model.fit(X)
    return model

def predict_lof(model, X):
    preds = model.predict(X)
    return (preds == -1).astype(int)
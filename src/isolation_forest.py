from sklearn.ensemble import IsolationForest

def train_isolation_forest(X):
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)
    return model

def predict_isolation(model, X):
    preds = model.predict(X)
    return (preds == -1).astype(int)
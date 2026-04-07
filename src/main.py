from preprocessing import load_data, preprocess
from zscore_model import zscore_anomaly
from isolation_forest import train_isolation_forest, predict_isolation
from lof_model import run_lof
from autoencoder import train_autoencoder, reconstruction_error
from threshold import find_best_threshold
from evaluation import evaluate

# Load data
df = load_data("./data/raw/creditcard.csv")
X, y = preprocess(df)

# Z-score
z_pred = zscore_anomaly(X)
print("\nZ-SCORE RESULTS")
evaluate(y, z_pred)

# Isolation Forest
iso_model = train_isolation_forest(X)
iso_pred = predict_isolation(iso_model, X)
print("\nISOLATION FOREST RESULTS")
evaluate(y, iso_pred)

# LOF
lof_pred = run_lof(X)
print("\nLOF RESULTS")
evaluate(y, lof_pred)

# Autoencoder
auto_model = train_autoencoder(X)
errors = reconstruction_error(auto_model, X)

threshold = find_best_threshold(y, errors)
auto_pred = (errors > threshold).astype(int)

print("\nAUTOENCODER RESULTS")
evaluate(y, auto_pred)
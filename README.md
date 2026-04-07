# 💳 Fraud Detection System — Anomaly Detection Pipeline

A modular fraud detection system built with Python that detects anomalous transactions using statistical, machine learning, and deep learning approaches. Includes an interactive **Streamlit web app** where users can upload transaction data and choose models to detect fraud in real time.

## Models Included
- **Z-Score** (Statistical baseline anomaly detection)
- **Isolation Forest** (Tree-based unsupervised anomaly detection)
- **Local Outlier Factor (LOF)** (Density-based anomaly detection)
- **Autoencoder (PyTorch)** (Deep learning reconstruction-based detection)

## Key Feature
- Interactive **Streamlit App**
  - Upload your own CSV transaction data
  - Select detection model
  - View fraud predictions instantly
  - Download results with predictions

## Project Structure
fraud-detection-system/
├── data/
│   ├── raw/                    # Original dataset (not tracked)
│   └── processed/              # Processed data (optional)
├── notebooks/
│   └── exploration.ipynb       # Exploratory Data Analysis
├── src/
│   ├── preprocessing.py        # Data preprocessing & scaling
│   ├── zscore_model.py         # Z-score anomaly detection
│   ├── isolation_forest.py     # Isolation Forest model
│   ├── lof_model.py            # LOF model (novelty detection)
│   ├── autoencoder.py          # PyTorch autoencoder
│   ├── threshold.py            # PR curve threshold tuning
│   ├── evaluation.py           # Model evaluation metrics
│   └── main.py                 # Run all models
├── app/
│   └── app.py                  # Streamlit application
├── models/                     # Saved models (optional)
├── reports/                    # Analysis/report files
├── requirements.txt
└── .gitignore

##  Dataset
- Source: Kaggle Credit Card Fraud Dataset  
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data  

###  Setup
Download the dataset and place it here:
data/raw/creditcard.csv

## ⚡ Quickstart
git clone https://github.com/Istiyak-Ishan/fraud-detection-system.git
cd fraud-detection-system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## ▶️ Run Project

### Run Full Pipeline
python src/main.py

### Run Streamlit App (Main Feature)
streamlit run app/app.py

Then open:
http://localhost:8501

## 📈 Features
- Multiple anomaly detection approaches (statistical + ML + DL)
- Threshold tuning using Precision-Recall curve
- Handles highly imbalanced fraud data
- Interactive UI for real-time usage
- Downloadable prediction results

## 🧠 Use Cases
- Credit card fraud detection  
- Financial anomaly detection  
- Transaction monitoring systems  
- Risk analysis pipelines  

## 🛠️ Tech Stack
Python · pandas · numpy · scikit-learn · PyTorch · Streamlit

## 📌 Future Improvements
- Model persistence (save/load trained models)
- SHAP explainability for fraud decisions
- Real-time API (FastAPI)
- Dashboard analytics (charts & insights)

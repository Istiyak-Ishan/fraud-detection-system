# 💳 Fraud Detection System — Anomaly Detection Pipeline

A modular fraud detection system using statistical, machine learning, and deep learning models. Includes an interactive **Streamlit app** to upload CSV data and detect anomalies in real time.

## Models
- Z-Score  
- Isolation Forest  
- Local Outlier Factor (LOF)  
- Autoencoder (PyTorch)  

## Main Feature
Upload CSV → Select model → Detect fraud → Download results via Streamlit UI.

## 📁 Project Structure
```
fraud-detection-system/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── zscore_model.py
│   ├── isolation_forest.py
│   ├── lof_model.py
│   ├── autoencoder.py
│   ├── threshold.py
│   ├── evaluation.py
│   └── main.py
├── app/
│   └── app.py
├── models/
├── reports/
├── requirements.txt
└── .gitignore
```

## 📊 Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data  
Place in: `data/raw/creditcard.csv`

## ⚡ Quickstart
```
git clone https://github.com/Istiyak-Ishan/fraud-detection-system.git
cd fraud-detection-system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run
```
python src/main.py
streamlit run app/app.py
```

## 🛠️ Stack
Python · pandas · numpy · scikit-learn · PyTorch · Streamlit

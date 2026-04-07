import streamlit as st
import pandas as pd
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import your modules
from preprocessing import preprocess
from isolation_forest import train_isolation_forest, predict_isolation
from lof_model import train_lof, predict_lof
from zscore_model import zscore_anomaly
from autoencoder import train_autoencoder, reconstruction_error
from threshold import find_best_threshold


st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection System")
st.write("Upload transaction data and detect fraud using multiple models.")



uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    
    model_choice = st.selectbox(
        "Select Model",
        ["Z-Score", "Isolation Forest", "LOF", "Autoencoder"]
    )

    
    if st.button("Run Detection"):

        st.info("Processing data...")

        # Preprocess
        X, y = preprocess(df)

      
        if model_choice == "Z-Score":
            preds = zscore_anomaly(X)

        elif model_choice == "Isolation Forest":
            model = train_isolation_forest(X)
            preds = predict_isolation(model, X)

        elif model_choice == "LOF":
            model = train_lof(X)
            preds = predict_lof(model, X)

        elif model_choice == "Autoencoder":
            model = train_autoencoder(X)
            errors = reconstruction_error(model, X)

            threshold = find_best_threshold(y, errors)
            preds = (errors > threshold).astype(int)

            df["Anomaly_Score"] = errors

        
        df["Fraud_Prediction"] = preds

        
        st.subheader("📈 Detection Results")
        st.dataframe(df.head())

        # Metrics
        total = len(df)
        frauds = int(preds.sum())
        normal = total - frauds

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Detected Fraud", frauds)
        col3.metric("Normal Transactions", normal)

        
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

        st.success("Detection Completed!")
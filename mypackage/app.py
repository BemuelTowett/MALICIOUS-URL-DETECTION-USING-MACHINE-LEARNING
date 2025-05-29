# streamlit run "C:\Users\HomePC\Documents\4th Year JKUAT\Project\Malicious URLs\mypackage\app.py"
import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from feature_extraction import FeatureExtractor

# ...existing code...
import re

# Function to validate URL using regex
def is_valid_url(url):
    regex = re.compile(
        r'^(https?://)?'  # http:// or https:// (optional)
        r'(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6})'  # domain
        r'(:\d+)?'  # optional port
        r'(/.*)?$'  # path
    )
    return re.match(regex, url) is not None

# Model paths and display names
model_files = {
    "Random Forest": "models/rf_model.pkl",
    "XGBoost": "models/xgb_c.pkl",
    "LightGBM": "models/lgb.pkl",
    "Neural Network": "models/neural_model.pkl",
    "RF Grid Search": "models/grid_search_rf_model.pkl"
}

# Streamlit UI
st.title("🔐 Malicious URL Detection")
st.write("Enter a URL below to check if it is safe or malicious using multiple models.")

url_input = st.text_input("Enter URL", "https://example.com")

if st.button("Predict"):
    if url_input:
        if not is_valid_url(url_input):
            st.warning("Please enter a valid URL (e.g., https://example.com)")
        else:
            try:
                # Extract features
                features = FeatureExtractor.extract(url_input)
                results = {}
                for model_name, path in model_files.items():
                    if os.path.exists(path):
                        model_obj = joblib.load(path)
                        if isinstance(model_obj, tuple):
                            model, feature_names = model_obj
                        else:
                            model = model_obj
                            feature_names = features.keys()
                        X = pd.DataFrame([features])[list(feature_names)]
                        pred_arr = model.predict(X)

                        # Fix: Safely extract scalar from prediction
                        if isinstance(pred_arr, (np.ndarray, list)) and len(pred_arr) > 0:
                            pred = int(pred_arr[0])
                        else:
                            pred = int(pred_arr)

                        # Handle label mapping
                        if hasattr(model, "classes_") and len(model.classes_) > 2:
                            label_map = {
                                0: "🟢 Benign",
                                1: "🔴 Defacement",
                                2: "🔴 Phishing",
                                3: "🔴 Malware"
                            }
                            label = label_map.get(pred, str(pred))
                        else:
                            label = "🔴 Malicious" if pred == 1 else "🟢 Safe"

                        results[model_name] = label
                    else:
                        results[model_name] = "Model file not found"
                st.subheader("Model Predictions")
                for model_name, label in results.items():
                    st.write(f"**{model_name}:** {label}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a URL.")

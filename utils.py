import pandas as pd
import joblib
import numpy as np

def load_model_and_scaler(model_path="model/svm_model.pkl", scaler_path="model/model_scaler.pkl"):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Model or scaler load failed: {e}")

def preprocess_input(df):
    if df.shape[1] != 30:
        raise ValueError("Input must have exactly 30 features.")
    if df.isnull().sum().sum() > 0:
        raise ValueError("Input contains missing values.")
    return df

def predict(model, scaler, input_df):
    scaled_input = scaler.transform(input_df)
    predictions = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)

    results = []
    for i in range(len(predictions)):
        pred_label = "Cancer" if predictions[i] == 0 else "No Cancer"
        confidence = float(np.max(probabilities[i]) * 100)

        results.append({
            "prediction": pred_label,
            "confidence": round(confidence, 2)
        })

    return results

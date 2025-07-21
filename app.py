import streamlit as st
import pandas as pd
from utils import load_model_and_scaler, preprocess_input, predict

st.set_page_config(page_title="CancerDetect - SVM Classifier", layout="centered")
st.title("Cancer Detection using SVM")
st.write("Upload a CSV file with 30 features to predict Cancer/No Cancer.")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(input_df.head())

        model, scaler = load_model_and_scaler()
        clean_df = preprocess_input(input_df)

        if st.button("Predict"):
            results = predict(model, scaler, clean_df)
            st.subheader("Prediction Results")
            for i, res in enumerate(results):
                st.markdown(f"**Sample {i+1}**")
                st.write(f"Prediction: `{res['prediction']}`")
                st.write(f"Confidence: `{res['confidence']}%`")
                st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a .csv file with 30 numeric feature columns.")

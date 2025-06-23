import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('enhanced_tuned_lightgbm_model.pkl')

# Streamlit UI
st.title("Customer Profitability Prediction")

uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # You must apply the same transformations here
    trained_features = ['age', 'income', 'gender_id', 'region_id']  # adjust as needed
    X_new = df[trained_features]

    preds = model.predict(X_new)

    df['prediction'] = preds
    st.write("📊 Prediction Results:")
    st.dataframe(df[['customer_id', 'prediction']])

    st.download_button("Download Results as Excel",
                       data=df.to_excel(index=False),
                       file_name="predictions.xlsx",
                       mime="application/vnd.ms-excel")

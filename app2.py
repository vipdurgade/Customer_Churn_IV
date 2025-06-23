import streamlit as st
import pandas as pd
import joblib
import io

# Load model
@st.cache_resource
def load_model():
    return joblib.load('enhanced_tuned_lightgbm_model.pkl')

model = load_model()

# Streamlit UI
st.title("Customer Profitability Prediction")
st.write("Upload an Excel file with the required features for prediction")

# Display required features
st.info("Required features in your Excel file:")
required_features = [
    "estimated_total_paid", 
    "carage_years", 
    "kosten_verw", 
    "kosten_prov", 
    "alter", 
    "KILOMETERSTAND_CLEAN", 
    "claim", 
    "state_id", 
    "plz_id", 
    "Cus_typ_id"
]
st.write(", ".join(required_features))

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("📋 Uploaded Data Preview:")
        st.dataframe(df.head())
        
        # Check if all required features are present
        missing_features = [feature for feature in required_features if feature not in df.columns]
        
        if missing_features:
            st.error(f"Missing required features: {', '.join(missing_features)}")
            st.write("Available columns in your file:")
            st.write(list(df.columns))
        else:
            # Extract features in the correct order
            X_new = df[required_features]
            
            # Check for missing values
            if X_new.isnull().any().any():
                st.warning("⚠️ Data contains missing values. Please clean your data first.")
                st.write("Missing values per column:")
                st.write(X_new.isnull().sum())
            else:
                # Make predictions
                with st.spinner('Making predictions...'):
                    predictions = model.predict(X_new)
                
                # Add predictions to dataframe
                df['prediction'] = predictions
                
                st.success("✅ Predictions completed!")
                st.write("📊 Prediction Results:")
                
                # Display results
                result_df = df.copy()
                st.dataframe(result_df)
                
                # Statistics
                st.write("📈 Prediction Statistics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(predictions))
                with col2:
                    st.metric("Average Prediction", f"{predictions.mean():.2f}")
                with col3:
                    st.metric("Prediction Range", f"{predictions.min():.2f} - {predictions.max():.2f}")
                
                # Create download button with proper Excel format
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Predictions')
                
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=output.getvalue(),
                    file_name="predictions_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure your Excel file is properly formatted and contains all required columns.")

# Add sidebar with information
st.sidebar.header("ℹ️ Model Information")
st.sidebar.write("**Model Type:** Enhanced Tuned LightGBM")
st.sidebar.write("**Features:** 10 input features")
st.sidebar.write("**Purpose:** Customer Profitability Prediction")

st.sidebar.header("📝 Instructions")
st.sidebar.write("""
1. Prepare Excel file with required features
2. Upload the file using the uploader
3. Review the predictions
4. Download results if satisfied
""")

# Footer
st.markdown("---")
st.markdown("*Model Version: enhanced_tuned_lightgbm_model.pkl*")
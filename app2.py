import streamlit as st
import pandas as pd
import joblib
import io
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"s
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background and font */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: none;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Card styling */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(45deg, #00b894, #00cec9);
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(45deg, #e17055, #d63031);
        border-radius: 10px;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #667eea;
        border-radius: 15px;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('lightgbm_model_vtr_weg_optuna100.pkl')
    except:
        st.error("Model file not found. Please ensure 'lightgbm_model_vtr_weg_optuna100.pkl' is in the same directory.")
        return None

model = load_model()

# Required features
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
    "Cus_typ_id",
    "vtr_dau"
]

# Feature descriptions for better user understanding
feature_descriptions = {
    "estimated_total_paid": "Total amount paid by customer (€)",
    "carage_years": "Years of car ownership",
    "kosten_verw": "Administrative costs (€)",
    "kosten_prov": "Provision costs (€)",
    "alter": "Customer age (years)",
    "KILOMETERSTAND_CLEAN": "Car mileage (km)",
    "claim": "Number of claims made",
    "state_id": "State identifier",
    "plz_id": "German postal code (PLZ)",
    "Cus_typ_id": "Customer type category",
    "vtr_dau": "Contract duration (days/years)"
}

# Customer type mapping
customer_types = {
    1: "Privatkunden",
    2: "Land- und Forstwirtschaft", 
    3: "Selbständige"
}

# State mapping
state_mapping = {
    1: "Brandenburg/Berlin/MV",
    2: "Hamburg/SH", 
    3: "Niedersachsen/Bremen",
    4: "NRW (partial)", 
    5: "Sachsen-Anhalt", 
    6: "NRW", 
    7: "Niedersachsen",
    8: "NRW", 
    9: "Rheinland-Pfalz", 
    10: "NRW", 
    11: "Hessen",
    12: "Saarland/RLP", 
    13: "Baden-Württemberg", 
    14: "Baden-Württemberg",
    15: "Bayern", 
    16: "Baden-Württemberg", 
    17: "Bayern/Thüringen"
}

# Main title
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced ML-powered customer analytics for better business decisions</p>', unsafe_allow_html=True)

if model is None:
    st.stop()

# File Upload Section
st.markdown('<div class="feature-card">', unsafe_allow_html=True)
st.markdown("### 📊 Bulk Prediction from File")
st.markdown("Upload your Excel or CSV file containing customer data for batch predictions.")

# Display required features in a nice format
with st.expander("📋 Required Features", expanded=True):
    col1, col2 = st.columns(2)
    for i, (feature, description) in enumerate(feature_descriptions.items()):
        if i % 2 == 0:
            col1.markdown(f"**{feature}**: {description}")
        else:
            col2.markdown(f"**{feature}**: {description}")

uploaded_file = st.file_uploader(
    "Choose your file", 
    type=['xlsx', 'xls', 'csv'],
    help="Upload an Excel or CSV file containing the required features"
)

if uploaded_file:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("✅ File uploaded successfully!")
        
        # Show data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Check if all required features are present
        missing_features = [feature for feature in required_features if feature not in df.columns]
        
        if missing_features:
            st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            st.info("Available columns in your file:")
            st.write(list(df.columns))
        else:
            # Extract features in the correct order
            X_new = df[required_features]
            
            # Check for missing values
            if X_new.isnull().any().any():
                st.warning("⚠️ Data contains missing values. Please clean your data first.")
                missing_summary = X_new.isnull().sum()
                missing_summary = missing_summary[missing_summary > 0]
                st.dataframe(missing_summary.to_frame("Missing Values"), use_container_width=True)
            else:
                # Make predictions
                if st.button("🚀 Generate Predictions", key="batch_predict"):
                    with st.spinner('🔮 Making predictions...'):
                        predictions = model.predict(X_new)
                        prediction_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Add predictions to dataframe
                    df['churn_prediction'] = predictions
                    if prediction_proba is not None:
                        df['churn_probability'] = prediction_proba
                    
                    st.success("✅ Predictions completed successfully!")
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Total Records", len(predictions))
                    with col2:
                        st.metric("📈 Churn Rate", f"{(predictions.sum() / len(predictions) * 100):.1f}%")
                    with col3:
                        st.metric("✅ Will Stay", f"{(predictions == 0).sum()}")
                    with col4:
                        st.metric("❌ Will Churn", f"{(predictions == 1).sum()}")
                    
                    # Show results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Create download button
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Predictions')
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=output.getvalue(),
                        file_name="churn_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your file is properly formatted and contains all required columns.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown("### ℹ️ Model Information")
    st.info("""
    **Model:** Enhanced Tuned LightGBM  
    **Purpose:** Customer Churn Prediction  
    **Features:** 11 input variables  
    **Accuracy:** Optimized for business use
    """)
    
    st.markdown("### ⚠️ Feature Encoding Check")
    st.warning("""
    **Important:** Ensure these match your training data:
    
    **Customer Types:**
    - 1: Privatkunden
    - 2: Land- und Forstwirtschaft  
    - 3: Selbständige
    
    **State Encoding:**
    - IDs 1-17 based on German states
    
    **PLZ Encoding:**
    - Using raw PLZ values
    - May need adjustment based on training
    
    **VTR DAU:**
    - Using input value as-is
    - Verify units (days/years) match training
    """)
    
    st.markdown("### 📚 How to Use")
    st.markdown("""
    **File Upload:**
    1. Prepare Excel/CSV with required features
    2. Upload using the file uploader
    3. Review predictions and download results
    """)
    
    st.markdown("### 🎯 Feature Importance")
    st.markdown("""
    Key factors affecting churn:
    - Total amount paid
    - Customer age
    - Car age and mileage
    - Claims history
    - Geographic location (PLZ/State)
    - Customer type category
    - Contract duration (VTR DAU)
    """)
    
    st.markdown("### 👥 Customer Types")
    for key, value in customer_types.items():
        st.markdown(f"**{key}.** {value}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>"
    "🔮 Powered by Advanced Machine Learning | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
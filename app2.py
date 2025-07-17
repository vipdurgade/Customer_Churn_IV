#st.markdown("### 💡 Model Features")
#st.markdown("""
#    - **Batch Processing**: Handle multiple customers at once
 #   - **Real-time Prediction**: Individual customer analysis
  #  - **Download Results**: Export predictions to Excel
   # - **Data Validation**: Check for missing or unknown values
    #- **Contract Duration**: New VTR DAU feature for better accuracy
    #""")
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
    initial_sidebar_state="expanded"
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #667eea;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white !important;
        border: none;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
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

# Load model and feature mapping
@st.cache_resource
def load_model():
    try:
        return joblib.load('lightgbm_model_vtr_weg_optuna100.pkl')
    except:
        st.error("Model file not found. Please ensure 'lightgbm_model_vtr_weg_optuna100.pkl' is in the same directory.")
        return None

@st.cache_resource
def load_feature_mapping():
    try:
        return joblib.load('factorize_mapping.pkl')
    except:
        st.error("Feature mapping file not found. Please ensure 'factorize_mapping.pkl' is in the same directory.")
        return None

model = load_model()
feature_mapping = load_feature_mapping()

# Feature encoding function
def encode_features(df, feature_mapping):
    """
    Encode categorical features using the saved mapping from training
    """
    if feature_mapping is None:
        st.error("Feature mapping not loaded. Cannot encode features.")
        return None
        
    df_encoded = df.copy()
    
    # Apply encoding for each feature that has a mapping
    for feature, mapping in feature_mapping.items():
        if feature in df_encoded.columns:
            # Convert the feature values using the mapping
            df_encoded[feature] = df_encoded[feature].map(mapping)
            
            # Handle unknown values
            if df_encoded[feature].isnull().any():
                unknown_values = df[feature][df_encoded[feature].isnull()].unique()
                st.warning(f"⚠️ Unknown values in {feature}: {unknown_values}")
                st.info(f"Available values for {feature}: {list(mapping.keys())}")
                
                # Option 1: Fill with most common value (mode)
                if len(mapping) > 0:
                    most_common_encoded = max(mapping.values(), key=list(mapping.values()).count)
                    df_encoded[feature].fillna(most_common_encoded, inplace=True)
                    st.info(f"Filled unknown values with most common encoding: {most_common_encoded}")
    
    return df_encoded

# Required features
required_features = [
    "estimated_total_paid",
    "vtr_dau",
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

# German postal code to state mapping
def get_state_from_plz(plz):
    """
    Convert German postal code (PLZ) to state ID
    Based on German postal code system
    """
    plz = int(plz)
    
    # German states with their postal code ranges
    if 1000 <= plz <= 19999:
        return 1  # Brandenburg, Berlin, Mecklenburg-Vorpommern
    elif 20000 <= plz <= 25999:
        return 2  # Hamburg, Schleswig-Holstein
    elif 26000 <= plz <= 31999:
        return 3  # Niedersachsen, Bremen
    elif 32000 <= plz <= 37999:
        return 4  # Nordrhein-Westfalen (partial)
    elif 38000 <= plz <= 39999:
        return 5  # Sachsen-Anhalt
    elif 40000 <= plz <= 48999:
        return 6  # Nordrhein-Westfalen
    elif 49000 <= plz <= 49999:
        return 7  # Niedersachsen (partial)
    elif 50000 <= plz <= 53999:
        return 8  # Nordrhein-Westfalen (partial)
    elif 54000 <= plz <= 56999:
        return 9  # Rheinland-Pfalz
    elif 57000 <= plz <= 59999:
        return 10  # Nordrhein-Westfalen (partial)
    elif 60000 <= plz <= 65999:
        return 11  # Hessen
    elif 66000 <= plz <= 68999:
        return 12  # Saarland, Rheinland-Pfalz (partial)
    elif 69000 <= plz <= 76999:
        return 13  # Baden-Württemberg
    elif 77000 <= plz <= 79999:
        return 14  # Baden-Württemberg (partial)
    elif 80000 <= plz <= 87999:
        return 15  # Bayern
    elif 88000 <= plz <= 89999:
        return 16  # Baden-Württemberg (partial)
    elif 90000 <= plz <= 96999:
        return 15  # Bayern (partial)
    elif 97000 <= plz <= 99999:
        return 17  # Bayern, Thüringen
    else:
        return 1  # Default fallback

# Get available options from feature mapping
def get_available_options(feature_name):
    """Get available options for a categorical feature from the mapping"""
    if feature_mapping and feature_name in feature_mapping:
        return list(feature_mapping[feature_name].keys())
    return []

# Reverse mapping for display purposes
def get_reverse_mapping(feature_name):
    """Get reverse mapping (encoded_value -> original_value) for display"""
    if feature_mapping and feature_name in feature_mapping:
        return {v: k for k, v in feature_mapping[feature_name].items()}
    return {}

# Feature descriptions for better user understanding
feature_descriptions = {
    "estimated_total_paid": "Total amount paid by customer (€)",
    "vtr_dau": "Contract duration (days)",
    "carage_years": "Years of car ownership",
    "kosten_verw": "Administrative costs (€)",
    "kosten_prov": "Provision costs (€)",
    "alter": "Customer age (years)",
    "KILOMETERSTAND_CLEAN": "Car mileage (km)",
    "claim": "Number of claims made",
    "state_id": "State identifier (auto-detected from PLZ)",
    "plz_id": "German postal code (PLZ)",
    "Cus_typ_id": "Customer type category"
}

# Main title
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced ML-powered customer analytics with automatic feature encoding</p>', unsafe_allow_html=True)

if model is None or feature_mapping is None:
    st.stop()

# Display feature mapping information
with st.expander("🔧 Feature Mapping Information", expanded=False):
    st.write("**Loaded Feature Mappings:**")
    for feature, mapping in feature_mapping.items():
        st.write(f"**{feature}:** {len(mapping)} unique values")
        # Show first few mappings as example
        sample_mapping = dict(list(mapping.items())[:5])
        st.write(f"  Example mappings: {sample_mapping}")
        if len(mapping) > 5:
            st.write(f"  ... and {len(mapping) - 5} more")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 File Upload", "✏️ Manual Input"])

with tab1:
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
            with st.expander("📋 Data Preview (Original)", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Check if all required features are present
            missing_features = [feature for feature in required_features if feature not in df.columns]
            
            if missing_features:
                st.error(f"❌ Missing required features: {', '.join(missing_features)}")
                st.info("Available columns in your file:")
                st.write(list(df.columns))
            else:
                # Extract features in the correct order
                X_new = df[required_features].copy()
                
                # Apply feature encoding
                st.info("🔧 Applying feature encoding...")
                X_encoded = encode_features(X_new, feature_mapping)
                
                if X_encoded is not None:
                    # Show encoded data preview
                    with st.expander("📋 Encoded Data Preview", expanded=False):
                        st.dataframe(X_encoded.head(10), use_container_width=True)
                    
                    # Check for missing values after encoding
                    if X_encoded.isnull().any().any():
                        st.warning("⚠️ Encoded data contains missing values. Please check the encoding process.")
                        missing_summary = X_encoded.isnull().sum()
                        missing_summary = missing_summary[missing_summary > 0]
                        st.dataframe(missing_summary.to_frame("Missing Values"), use_container_width=True)
                    else:
                        # Make predictions
                        if st.button("🚀 Generate Predictions", key="batch_predict"):
                            with st.spinner('🔮 Making predictions...'):
                                predictions = model.predict(X_encoded)
                                prediction_proba = model.predict_proba(X_encoded)[:, 1] if hasattr(model, 'predict_proba') else None
                            
                            # Add predictions to original dataframe
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

with tab2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### ✏️ Single Customer Prediction")
    st.markdown("Enter customer details manually for individual prediction.")
    
    # Create input form
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            estimated_total_paid = st.number_input(
                "💰 Estimated Total Paid (€)", 
                min_value=0.0, 
                value=1000.0,
                help="Total amount paid by the customer"
            )
            
            vtr_dau = st.number_input(
                "📅 Contract Duration (Days)", 
                min_value=0, 
                value=365,
                help="Duration of the contract in days"
            )
            
            carage_years = st.number_input(
                "🚗 Car Age (Years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=5.0,
                help="How many years the customer has owned the car"
            )
            
            kosten_verw = st.number_input(
                "📋 Administrative Costs (€)", 
                min_value=0.0, 
                value=100.0,
                help="Administrative costs associated with the customer"
            )
            
            kosten_prov = st.number_input(
                "💼 Provision Costs (€)", 
                min_value=0.0, 
                value=50.0,
                help="Provision costs for the customer"
            )
            
            alter = st.number_input(
                "👤 Customer Age", 
                min_value=18, 
                max_value=100, 
                value=35,
                help="Age of the customer"
            )
        
        with col2:
            kilometerstand = st.number_input(
                "🛣️ Car Mileage (km)", 
                min_value=0, 
                value=50000,
                help="Current mileage of the car"
            )
            
            claim = st.number_input(
                "📊 Number of Claims", 
                min_value=0, 
                max_value=20, 
                value=0,
                help="Number of insurance claims made"
            )
            
            plz_id = st.number_input(
                "📮 German Postal Code (PLZ)", 
                min_value=1000, 
                max_value=99999,
                value=10115,
                help="Enter German postal code (PLZ) - state will be detected automatically"
            )
            
            # Auto-detect state from PLZ
            detected_state = get_state_from_plz(plz_id)
            
            # Customer type selection based on available options in mapping
            customer_type_options = get_available_options('Cus_typ_id')
            if customer_type_options:
                customer_type_display = st.selectbox(
                    "👥 Customer Type", 
                    options=customer_type_options,
                    index=0,
                    help="Select the customer category"
                )
            else:
                # Fallback to default options if mapping not available
                customer_type_display = st.selectbox(
                    "👥 Customer Type", 
                    options=["Privatkunden", "Land- und Forstwirtschaft", "Selbständige"],
                    index=0,
                    help="Select the customer category"
                )
            
            st.info(f"🗺️ **Auto-detected State ID:** {detected_state}")
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'estimated_total_paid': [estimated_total_paid],
                'vtr_dau': [vtr_dau],
                'carage_years': [carage_years],
                'kosten_verw': [kosten_verw],
                'kosten_prov': [kosten_prov],
                'alter': [alter],
                'KILOMETERSTAND_CLEAN': [kilometerstand],
                'claim': [claim],
                'state_id': [detected_state],
                'plz_id': [plz_id],
                'Cus_typ_id': [customer_type_display]
            })
            
            # Display original input data
            with st.expander("📋 Original Input Data", expanded=False):
                st.dataframe(input_data)
            
            # Apply feature encoding
            input_data_encoded = encode_features(input_data, feature_mapping)
            
            if input_data_encoded is not None:
                # Display encoded data for debugging
                with st.expander("🔧 Encoded Input Data", expanded=False):
                    st.dataframe(input_data_encoded)
                
                # Make prediction
                with st.spinner('🔮 Analyzing customer data...'):
                    prediction = model.predict(input_data_encoded)[0]
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(input_data_encoded)[0, 1]
                    else:
                        probability = None
                
                # Display results
                st.markdown("---")
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("❌ **High Churn Risk**")
                        st.markdown("This customer is likely to churn.")
                    else:
                        st.success("✅ **Low Churn Risk**")
                        st.markdown("This customer is likely to stay.")
                
                with col2:
                    if probability is not None:
                        st.metric("🎯 Churn Probability", f"{probability:.2%}")
                        
                        # Risk level based on probability
                        if probability < 0.3:
                            risk_level = "🟢 Low Risk"
                        elif probability < 0.7:
                            risk_level = "🟡 Medium Risk"
                        else:
                            risk_level = "🔴 High Risk"
                        
                        st.markdown(f"**Risk Level:** {risk_level}")
                
                with col3:
                    st.metric("📊 Prediction", "Churn" if prediction == 1 else "Stay")
                    confidence = max(probability, 1-probability) if probability is not None else 0.5
                    st.metric("🎯 Confidence", f"{confidence:.2%}")
                
                # Recommendation based on prediction
                if prediction == 1:
                    st.markdown("### 💡 Recommendations")
                    st.warning("""
                    **Action Required:** This customer shows high churn risk. Consider:
                    - Proactive outreach and engagement
                    - Special offers or incentives
                    - Personalized retention campaigns
                    - Customer satisfaction surveys
                    """)
                else:
                    st.markdown("### 💡 Recommendations")
                    st.info("""
                    **Good News:** This customer shows low churn risk. Consider:
                    - Maintaining current service quality
                    - Cross-selling opportunities
                    - Loyalty program enrollment
                    - Regular check-ins
                    """)
            else:
                st.error("❌ Failed to encode input data. Please check the feature mapping.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown("### ℹ️ Model Information")
    st.info("""
    **Model:** LightGBM VTR WEG Optuna100  
    **Purpose:** Customer Churn Prediction  
    **Features:** 11 input variables  
    **Encoding:** Automatic via factorize_mapping.pkl
    """)
    
    if feature_mapping:
        st.markdown("### 🔧 Feature Encoding Status")
        st.success("✅ Feature mapping loaded successfully!")
        st.write(f"**Mapped Features:** {len(feature_mapping)}")
        
        # Show which features have mappings
        for feature in required_features:
            if feature in feature_mapping:
                st.write(f"✅ {feature}: {len(feature_mapping[feature])} categories")
            else:
                st.write(f"⚪ {feature}: Numerical (no mapping needed)")
    else:
        st.markdown("### ⚠️ Feature Encoding Status")
        st.error("❌ Feature mapping not loaded!")
    
    st.markdown("### 📚 How to Use")
    st.markdown("""
    **File Upload:**
    1. Prepare Excel/CSV with required features
    2. Upload using the file uploader
    3. Features will be automatically encoded
    4. Review predictions and download results
    
    **Manual Input:**
    1. Fill in customer details in the form
    2. Click 'Predict Churn' button
    3. Data will be automatically encoded
    4. Review individual prediction results
    """)
    
    st.markdown("### 🎯 Key Features")
    st.markdown("""
    Key factors affecting churn:
    - Total amount paid
    - Contract duration (VTR DAU)
    - Customer age
    - Car age and mileage
    - Claims history
    - Geographic location (PLZ/State)
    - Customer type category
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>"
    "🔮 Powered by Advanced Machine Learning with Automatic Feature Encoding | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
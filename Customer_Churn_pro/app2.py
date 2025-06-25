import streamlit as st
import pandas as pd
import joblib
import io
import numpy as np
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction - Itzehoer Versicherungen",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and encode image
def get_base64_encoded_image(image_path):
    """Convert image to base64 string for embedding in CSS"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Custom CSS for Itzehoer Versicherungen styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Open+Sans:wght@300;400;600;700&display=swap');
    
    /* Main background - Itzehoer inspired */
    .stApp {
        background: linear-gradient(135deg, #7CB342 0%, #558B2F 50%, #33691E 100%);
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Header section with logo */
    .header-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 0 0 25px 25px;
        padding: 1.5rem 2rem;
        margin: 0 -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border-bottom: 4px solid #7CB342;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .company-logo {
        height: 80px;
        margin-right: 20px;
    }
    
    /* Main content container */
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(124, 179, 66, 0.2);
    }
    
    /* Title styling - Itzehoer colors */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2E3842 !important;
        text-align: center;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .company-name {
        font-size: 1.1rem;
        color: #7CB342;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling - Itzehoer inspired */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(124, 179, 66, 0.1);
        border-left: 4px solid #7CB342;
    }
    
    /* Button styling - Itzehoer green */
    .stButton > button {
        background: linear-gradient(45deg, #7CB342, #558B2F);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 179, 66, 0.3);
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #558B2F, #33691E);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124, 179, 66, 0.4);
        border: 2px solid #7CB342;
    }
    
    /* Sidebar styling - Itzehoer theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e8f5e8 100%);
    }
    
    .css-1lcbmhc {
        background: linear-gradient(180deg, #f8f9fa 0%, #e8f5e8 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(124, 179, 66, 0.1);
        border-top: 4px solid #7CB342;
    }
    
    /* Tab styling - Itzehoer colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(124, 179, 66, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #558B2F;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #7CB342, #558B2F);
        color: white !important;
        border: 2px solid #7CB342;
    }
    
    /* Input styling - Itzehoer theme */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e8f5e8;
        transition: border-color 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #7CB342;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e8f5e8;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #7CB342;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(45deg, #7CB342, #558B2F);
        border-radius: 10px;
        color: white;
    }
    
    .stError {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #f39c12, #e67e22);
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(45deg, #3498db, #2980b9);
        border-radius: 10px;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #7CB342;
        border-radius: 15px;
        background: rgba(124, 179, 66, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #558B2F;
        background: rgba(124, 179, 66, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(124, 179, 66, 0.1);
        border-radius: 10px;
        color: #2E3842;
        font-weight: 600;
    }
    
    /* Form styling */
    .stForm {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(124, 179, 66, 0.1);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Sidebar info boxes */
    .sidebar-info {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #7CB342;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid rgba(124, 179, 66, 0.2);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #7CB342;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('enhanced_tuned_lightgbm_model.pkl')
    except:
        st.error("Model file not found. Please ensure 'enhanced_tuned_lightgbm_model.pkl' is in the same directory.")
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

# PLZ encoding function
def encode_plz(plz):
    """
    Encode PLZ to match training data
    """
    return int(plz)

# Feature descriptions for better user understanding
feature_descriptions = {
    "estimated_total_paid": "Total amount paid by customer (€)",
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

# Header with logo section
st.markdown('<div class="header-container">', unsafe_allow_html=True)

# Logo section - placeholder for now, will be replaced when logo is uploaded
st.markdown('''
<div class="logo-section">
    <div style="width: 80px; height: 80px; background: linear-gradient(45deg, #7CB342, #558B2F); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-right: 20px;">
        <span style="color: white; font-size: 2rem; font-weight: bold;">I</span>
    </div>
    <div>
        <div class="company-name">ITZEHOER VERSICHERUNGEN</div>
        <div class="main-title">🎯 Customer Churn Prediction</div>
    </div>
</div>
''', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Advanced ML-powered customer analytics for better business decisions</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if model is None:
    st.stop()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

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
                            file_name="itzehoer_churn_predictions.xlsx",
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
                value=25524,  # Itzehoer PLZ
                help="Enter German postal code (PLZ) - state will be detected automatically"
            )
            
            # Auto-detect state from PLZ
            detected_state = get_state_from_plz(plz_id)
            
            st.info(f"🗺️ **Auto-detected State:** {state_mapping.get(detected_state, 'Unknown')} (ID: {detected_state})")
            
            # Customer type selection with meaningful labels
            customer_type_display = st.selectbox(
                "👥 Customer Type", 
                options=list(customer_types.keys()),
                format_func=lambda x: f"{customer_types[x]}",
                index=0,
                help="Select the customer category"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
        
        if submitted:
            # Prepare input data with proper encoding
            input_data = pd.DataFrame({
                'estimated_total_paid': [estimated_total_paid],
                'carage_years': [carage_years],
                'kosten_verw': [kosten_verw],
                'kosten_prov': [kosten_prov],
                'alter': [alter],
                'KILOMETERSTAND_CLEAN': [kilometerstand],
                'claim': [claim],
                'state_id': [detected_state],
                'plz_id': [encode_plz(plz_id)],
                'Cus_typ_id': [customer_type_display]
            })
            
            # Display the data being sent to model for debugging
            with st.expander("🔧 Debug: Data sent to model", expanded=False):
                st.dataframe(input_data)
            
            # Make prediction
            with st.spinner('🔮 Analyzing customer data...'):
                prediction = model.predict(input_data)[0]
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)[0, 1]
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
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Model Information")
    st.info("""
    **Model:** Enhanced Tuned LightGBM  
    **Purpose:** Customer Churn Prediction  
    **Features:** 10 input variables  
    **Accuracy:** Optimized for business use
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 📚 How to Use")
    st.markdown("""
    **File Upload:**
    1. Prepare Excel/CSV with required features
    2. Upload using the file uploader
    3. Review predictions and download results
    
    **Manual Input:**
    1. Fill in customer details in the form
    2. Click 'Predict Churn' button
    3. Review individual prediction results
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 🎯 Feature Importance")
    st.markdown("""
    Key factors affecting churn:
    - Total amount paid
    - Customer age
    - Car age and mileage
    - Claims history
    - Geographic location (PLZ/State)
    - Customer type category
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 🗺️ State Detection")
    st.markdown("""
    **Automatic state detection** from German postal codes:
    - 10000-19999: Brandenburg/Berlin/MV
    - 20000-25999: Hamburg/Schleswig-Holstein
    - 26000-31999: Niedersachsen/Bremen
    - 32000-48999: Nordrhein-Westfalen
    - 60000-65999: Hessen
    - 80000-96999: Bayern
    - And more...
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("### 👥")
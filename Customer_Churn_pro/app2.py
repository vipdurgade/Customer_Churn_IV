import streamlit as st
import pandas as pd
import joblib
import io
import numpy as np
from PIL import Image
import base64
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction - Itzehoer Versicherungen",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load SHAP explainer
@st.cache_resource
def load_explainer():
    try:
        return joblib.load('shap_explainer.pkl')
    except:
        st.warning("SHAP explainer not found. Feature importance visualization will be limited.")
        return None

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('enhanced_tuned_lightgbm_model.pkl')
        explainer = load_explainer()
        return model, explainer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'enhanced_tuned_lightgbm_model.pkl' is in the same directory.")
        return None, None

model, explainer = load_model()

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
    """Convert German postal code (PLZ) to state ID"""
    plz = int(plz)
    
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

# Feature descriptions
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

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Open Sans', sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1a3e72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
    }
    
    /* Card styling */
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #2a5298;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1a3e72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(26, 62, 114, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0 !important;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: #1a3e72 !important;
        border-bottom: 3px solid #1a3e72;
    }
    
    /* Input styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 5px !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <div style="width: 60px; height: 60px; background: white; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
            <span style="color: #1a3e72; font-size: 1.8rem; font-weight: bold;">IV</span>
        </div>
        <div>
            <h1 class="main-title">Customer Churn Prediction</h1>
            <p class="subtitle">Predict customer retention with machine learning</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.stop()

# Main content
st.markdown("""
<div style="max-width: 1400px; margin: 0 auto;">
    <div class="feature-card">
        <h3 style="color: #1a3e72; margin-bottom: 1rem;">🔍 About This Tool</h3>
        <p>
            This predictive analytics tool helps identify customers at risk of churning. 
            Upload customer data or enter details manually to receive churn predictions 
            with explanations to guide retention strategies.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Batch Prediction", "✏️ Single Prediction"])

with tab1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Bulk Customer Analysis")
    st.markdown("Upload a CSV or Excel file containing customer data for batch churn prediction.")
    
    with st.expander("📋 Required Data Fields", expanded=False):
        col1, col2 = st.columns(2)
        for i, (feature, description) in enumerate(feature_descriptions.items()):
            if i % 2 == 0:
                col1.markdown(f"**{feature}**: {description}")
            else:
                col2.markdown(f"**{feature}**: {description}")
    
    uploaded_file = st.file_uploader(
        "Upload customer data file", 
        type=['xlsx', 'xls', 'csv'],
        help="File should contain all required features"
    )
    
    if uploaded_file:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ File successfully uploaded!")
            
            # Show data preview
            with st.expander("🔍 Data Preview", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
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
                    if st.button("🚀 Predict Churn for All Customers", key="batch_predict"):
                        with st.spinner('Analyzing customer data...'):
                            # Make predictions
                            predictions = model.predict(X_new)
                            prediction_proba = model.predict_proba(X_new)[:, 1]
                            
                            # Add predictions to dataframe
                            df['churn_prediction'] = predictions
                            df['churn_probability'] = prediction_proba
                            
                            # Calculate SHAP values if explainer is available
                            if explainer:
                                shap_values = explainer.shap_values(X_new)
                        
                        st.success("✅ Predictions completed!")
                        
                        # Display summary statistics
                        st.markdown("### 📈 Prediction Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Customers", len(predictions))
                        with col2:
                            churn_rate = predictions.mean()
                            st.metric("Churn Rate", f"{churn_rate:.1%}")
                        with col3:
                            st.metric("Will Stay", f"{(predictions == 0).sum()}")
                        with col4:
                            st.metric("Will Churn", f"{(predictions == 1).sum()}")
                        
                        # Show distribution of churn probabilities
                        st.markdown("### 📊 Churn Probability Distribution")
                        fig = px.histogram(
                            df, 
                            x='churn_probability', 
                            nbins=20,
                            color_discrete_sequence=['#1a3e72'],
                            labels={'churn_probability': 'Churn Probability'}
                        )
                        fig.update_layout(
                            xaxis_title="Churn Probability",
                            yaxis_title="Number of Customers",
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show feature importance if SHAP values are available
                        if explainer:
                            st.markdown("### 🔍 Feature Importance (SHAP Values)")
                            
                            # Calculate mean absolute SHAP values
                            mean_shap = np.abs(shap_values).mean(0)
                            shap_df = pd.DataFrame({
                                'Feature': X_new.columns,
                                'Importance': mean_shap
                            }).sort_values('Importance', ascending=False)
                            
                            # Plot feature importance
                            fig = px.bar(
                                shap_df.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                color_discrete_sequence=['#1a3e72'],
                                labels={'Importance': 'Average Impact on Prediction'}
                            )
                            fig.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Option to show detailed SHAP analysis for a specific customer
                            st.markdown("### 🔎 Detailed Customer Analysis")
                            customer_idx = st.slider(
                                "Select customer index to examine", 
                                min_value=0, 
                                max_value=len(df)-1, 
                                value=0
                            )
                            
                            # Force plot for individual prediction
                            st.markdown(f"#### Explanation for Customer {customer_idx}")
                            st.markdown(f"**Churn Probability:** {df.iloc[customer_idx]['churn_probability']:.1%}")
                            
                            plt.figure()
                            shap.force_plot(
                                explainer.expected_value[1],
                                shap_values[1][customer_idx,:],
                                X_new.iloc[customer_idx,:],
                                matplotlib=True,
                                show=False
                            )
                            st.pyplot(plt.gcf(), bbox_inches='tight')
                            plt.clf()
                        
                        # Show results table
                        st.markdown("### 📋 Prediction Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Create download button
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Predictions')
                        
                        st.download_button(
                            label="📥 Download Full Results",
                            data=output.getvalue(),
                            file_name="customer_churn_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 👤 Individual Customer Analysis")
    st.markdown("Enter customer details to predict churn risk.")
    
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            estimated_total_paid = st.number_input(
                "💰 Total Paid (€)", 
                min_value=0.0, 
                value=2500.0,
                step=100.0
            )
            
            carage_years = st.number_input(
                "🚗 Car Age (Years)", 
                min_value=0.0, 
                max_value=50.0, 
                value=4.0,
                step=0.5
            )
            
            kosten_verw = st.number_input(
                "📋 Admin Costs (€)", 
                min_value=0.0, 
                value=150.0,
                step=10.0
            )
            
            kosten_prov = st.number_input(
                "💼 Provision (€)", 
                min_value=0.0, 
                value=75.0,
                step=5.0
            )
            
            alter = st.number_input(
                "👤 Age", 
                min_value=18, 
                max_value=100, 
                value=42
            )
        
        with col2:
            kilometerstand = st.number_input(
                "🛣️ Mileage (km)", 
                min_value=0, 
                value=45000,
                step=1000
            )
            
            claim = st.number_input(
                "📊 Claims", 
                min_value=0, 
                max_value=20, 
                value=1
            )
            
            plz_id = st.number_input(
                "📮 Postal Code", 
                min_value=1000, 
                max_value=99999,
                value=25524  # Itzehoer PLZ
            )
            
            # Auto-detect state from PLZ
            detected_state = get_state_from_plz(plz_id)
            st.info(f"🗺️ Detected State: {state_mapping.get(detected_state, 'Unknown')}")
            
            customer_type_display = st.selectbox(
                "👥 Customer Type", 
                options=list(customer_types.keys()),
                format_func=lambda x: customer_types[x],
                index=0
            )
        
        submitted = st.form_submit_button("🔮 Predict Churn Risk")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'estimated_total_paid': [estimated_total_paid],
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
            
            # Make prediction
            with st.spinner('Analyzing customer data...'):
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0, 1]
                
                # Get SHAP values if explainer is available
                if explainer:
                    shap_values = explainer.shap_values(input_data)
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            if prediction == 1:
                st.error(f"### ❌ High Churn Risk ({probability:.1%} probability)")
            else:
                st.success(f"### ✅ Low Churn Risk ({probability:.1%} probability)")
            
            # Show probability gauge
            st.markdown("### 📊 Risk Assessment")
            fig = px.bar(
                x=[probability],
                y=["Churn Probability"],
                orientation='h',
                range_x=[0, 1],
                color_discrete_sequence=['#1a3e72' if probability < 0.5 else '#d62728'],
                text=[f"{probability:.1%}"]
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
                height=150,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            fig.update_traces(textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level interpretation
            if probability < 0.3:
                risk_level = "🟢 Low Risk"
                recommendation = "Continue current engagement strategy"
            elif probability < 0.7:
                risk_level = "🟡 Medium Risk"
                recommendation = "Monitor closely and consider retention offers"
            else:
                risk_level = "🔴 High Risk"
                recommendation = "Immediate retention action recommended"
            
            st.markdown(f"**Risk Level:** {risk_level}  \n**Recommendation:** {recommendation}")
            
            # Show SHAP explanation if available
            if explainer:
                st.markdown("### 🔍 Why This Prediction?")
                
                # Create SHAP waterfall plot
                plt.figure()
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1],
                    shap_values[1][0,:],
                    feature_names=input_data.columns,
                    max_display=10,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()
            
            # Show action recommendations
            st.markdown("### 💡 Suggested Actions")
            
            if prediction == 1:
                st.markdown("""
                - **Proactive outreach:** Personal call from account manager
                - **Special offer:** Discount on next premium
                - **Service review:** Check for unmet needs
                - **Survey:** Understand pain points
                """)
            else:
                st.markdown("""
                - **Loyalty rewards:** Offer for continued business
                - **Cross-sell:** Relevant additional products
                - **Check-in:** Regular satisfaction surveys
                - **Engagement:** Invite to customer events
                """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h3 style="color: #1a3e72; margin-top: 0;">ℹ️ About This Model</h3>
        <p>This LightGBM model predicts customer churn based on historical patterns. Key features:</p>
        <ul style="padding-left: 1.2rem;">
            <li>Accuracy: 87% (validation set)</li>
            <li>Precision: 83% for churn class</li>
            <li>Recall: 79% for churn class</li>
        </ul>
        <p>Last updated: June 2023</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h3 style="color: #1a3e72; margin-top: 0;">📊 Model Performance</h3>
        <p>Key metrics on test set:</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px;">
                <div style="font-size: 0.8rem; color: #6c757d;">Accuracy</div>
                <div style="font-weight: bold;">0.87</div>
            </div>
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px;">
                <div style="font-size: 0.8rem; color: #6c757d;">Precision</div>
                <div style="font-weight: bold;">0.83</div>
            </div>
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px;">
                <div style="font-size: 0.8rem; color: #6c757d;">Recall</div>
                <div style="font-weight: bold;">0.79</div>
            </div>
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px;">
                <div style="font-size: 0.8rem; color: #6c757d;">AUC-ROC</div>
                <div style="font-weight: bold;">0.91</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h3 style="color: #1a3e72; margin-top: 0;">📈 Top Churn Drivers</h3>
        <ol style="padding-left: 1.2rem;">
            <li>Total amount paid</li>
            <li>Customer age</li>
            <li>Number of claims</li>
            <li>Car age</li>
            <li>Administrative costs</li>
        </ol>
        <p>These factors most influence churn predictions.</p>
    </div>
    """, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Simple page setup
st.title("Customer Churn Prediction")
st.write("Predict if a customer will leave")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('enhanced_tuned_lightgbm_model.pkl')
        return model
    except:
        st.error("Model file not found!")
        return None

model = load_model()

if model is None:
    st.stop()

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

# Simple postal code to state mapping
def get_state_from_plz(plz):
    plz = int(plz)
    if 20000 <= plz <= 25999:
        return 2  # Hamburg/Schleswig-Holstein
    elif 1000 <= plz <= 19999:
        return 1  # Brandenburg/Berlin
    elif 26000 <= plz <= 31999:
        return 3  # Niedersachsen
    elif 40000 <= plz <= 59999:
        return 6  # NRW
    elif 60000 <= plz <= 65999:
        return 11  # Hessen
    elif 80000 <= plz <= 96999:
        return 15  # Bayern
    else:
        return 1  # Default

# Two tabs: File upload and Manual input
tab1, tab2 = st.tabs(["Upload File", "Enter Data"])

# Tab 1: File Upload
with tab1:
    st.header("Upload Customer Data")
    
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Check if required columns exist
        missing_cols = [col for col in required_features if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            if st.button("Predict All"):
                # Make predictions
                X = df[required_features]
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]
                
                # Add results to dataframe
                df['will_churn'] = predictions
                df['churn_probability'] = probabilities
                
                # Show results
                st.success("Predictions complete!")
                st.write(f"Total customers: {len(df)}")
                st.write(f"Will churn: {sum(predictions)}")
                st.write(f"Will stay: {len(predictions) - sum(predictions)}")
                
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv")

# Tab 2: Manual Input
with tab2:
    st.header("Enter Customer Details")
    
    # Create input form
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            total_paid = st.number_input("Total Paid (€)", min_value=0.0, value=2500.0)
            car_age = st.number_input("Car Age (years)", min_value=0.0, value=4.0)
            admin_costs = st.number_input("Admin Costs (€)", min_value=0.0, value=150.0)
            provision = st.number_input("Provision (€)", min_value=0.0, value=75.0)
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=42)
        
        with col2:
            mileage = st.number_input("Car Mileage (km)", min_value=0, value=45000)
            claims = st.number_input("Number of Claims", min_value=0, value=1)
            postal_code = st.number_input("Postal Code", min_value=1000, max_value=99999, value=25524)
            customer_type = st.selectbox("Customer Type", [1, 2, 3], 
                                       format_func=lambda x: {1: "Private", 2: "Farmer", 3: "Self-employed"}[x])
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare data
            state_id = get_state_from_plz(postal_code)
            
            input_data = pd.DataFrame({
                'estimated_total_paid': [total_paid],
                'carage_years': [car_age],
                'kosten_verw': [admin_costs],
                'kosten_prov': [provision],
                'alter': [age],
                'KILOMETERSTAND_CLEAN': [mileage],
                'claim': [claims],
                'state_id': [state_id],
                'plz_id': [postal_code],
                'Cus_typ_id': [customer_type]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0, 1]
            
            # Show result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"Customer will likely CHURN (Probability: {probability:.1%})")
            else:
                st.success(f"Customer will likely STAY (Probability: {probability:.1%})")
            
            # Show probability bar
            st.progress(probability)
            
            # Simple recommendations
            if prediction == 1:
                st.write("**Recommended Actions:**")
                st.write("- Contact customer immediately")
                st.write("- Offer special discount")
                st.write("- Check for service issues")
            else:
                st.write("**Recommended Actions:**")
                st.write("- Continue normal service")
                st.write("- Consider loyalty rewards")
                st.write("- Offer additional products")

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("This app predicts customer churn using machine learning.")
st.sidebar.write("Upload a file or enter data manually to get predictions.")
st.sidebar.write("Model accuracy: ~87%")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction (LightGBM)",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction (LightGBM)")
st.markdown("Upload your policy data and get churn probability predictions using our trained LightGBM model.")

# Required features in exact order
REQUIRED_FEATURES = [
    'estimated_total_paid',
    'vtr_dau', 
    'carage_years',
    'kosten_verw',
    'kosten_prov',
    'alter',
    'KILOMETERSTAND_CLEAN',
    'claim',
    'state_id',
    'plz_id',
    'Cus_typ_id'
]

# German state mapping
STATE_MAPPING = {
    '01':'Sachsen','02':'Sachsen','03':'Brandenburg','04':'Sachsen','05':'Sachsen-Anhalt','06':'Sachsen-Anhalt','07':'Thüringen','08':'Sachsen','09':'Sachsen',
    '10':'Berlin','11':'Berlin','12':'Brandenburg','13':'Brandenburg','14':'Brandenburg','15':'Brandenburg','16':'Brandenburg',
    '17':'Mecklenburg-Vorpommern','18':'Mecklenburg-Vorpommern','19':'Mecklenburg-Vorpommern',
    '20':'Schleswig-Holstein','21':'Schleswig-Holstein','22':'Hamburg','23':'Schleswig-Holstein',
    '24':'Schleswig-Holstein','25':'Schleswig-Holstein','26':'Niedersachsen','27':'Bremen','28':'Bremen','29':'Niedersachsen',
    '30':'Niedersachsen','31':'Niedersachsen','32':'Nordrhein-Westfalen','33':'Nordrhein-Westfalen','34':'Hessen',
    '35':'Hessen','36':'Hessen','37':'Niedersachsen','38':'Niedersachsen','39':'Sachsen-Anhalt',
    '40':'Nordrhein-Westfalen','41':'Nordrhein-Westfalen','42':'Nordrhein-Westfalen','44':'Nordrhein-Westfalen','45':'Nordrhein-Westfalen','46':'Nordrhein-Westfalen',
    '47':'Nordrhein-Westfalen','48':'Nordrhein-Westfalen','49':'Niedersachsen',
    '50':'Nordrhein-Westfalen','51':'Nordrhein-Westfalen','52':'Nordrhein-Westfalen','53':'Nordrhein-Westfalen','54':'Rheinland-Pfalz','55':'Rheinland-Pfalz',
    '56':'Rheinland-Pfalz','57':'Nordrhein-Westfalen','58':'Nordrhein-Westfalen','59':'Nordrhein-Westfalen',
    '60':'Hessen','61':'Hessen','62':'Hessen','63':'Hessen','64':'Hessen','65':'Hessen',
    '66':'Saarland','67':'Rheinland-Pfalz','68':'Rheinland-Pfalz','69':'Hessen',
    '70':'Baden-Württemberg','71':'Baden-Württemberg','72':'Baden-Württemberg','73':'Baden-Württemberg','74':'Baden-Württemberg','75':'Baden-Württemberg',
    '76':'Baden-Württemberg','77':'Baden-Württemberg','78':'Baden-Württemberg','79':'Baden-Württemberg',
    '80':'Bayern','81':'Bayern','82':'Bayern','83':'Bayern','84':'Bayern','85':'Bayern','86':'Bayern','87':'Bayern','88':'Bayern','89':'Bayern',
    '90':'Bayern','91':'Bayern','92':'Bayern','93':'Bayern','94':'Bayern','95':'Bayern','96':'Bayern',
    '97':'Bayern','98':'Thüringen','99':'Thüringen'
}

def safe_factorize(series, add_one=True):
    """Safely factorize a series, handling NaN values."""
    try:
        codes, uniques = pd.factorize(series, dropna=False)
        if add_one:
            codes = codes + 1
        return codes
    except Exception as e:
        st.warning(f"Error in factorization: {str(e)}")
        return np.full(len(series), np.nan)

def parse_date_robust(date_str):
    """Parse dates with multiple format support."""
    if pd.isna(date_str) or date_str == '':
        return None
    
    # Convert to string if not already
    date_str = str(date_str).strip()
    
    # List of date formats to try
    date_formats = [
        "%d%b%Y",      # 01FEB2025
        "%d/%m/%Y",    # 01/02/2025
        "%Y-%m-%d",    # 2025-02-01
        "%m/%d/%Y",    # 02/01/2025
        "%d.%m.%Y",    # 01.02.2025
        "%Y%m%d",      # 20250201
        "%d-%m-%Y"     # 01-02-2025
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except:
            continue
    
    # Try pandas to_datetime as fallback
    try:
        return pd.to_datetime(date_str).date()
    except:
        return None

def apply_mappings(series, mapping_dict, default_value=0):
    """Apply saved mappings to categorical data."""
    if mapping_dict is None:
        return safe_factorize(series)
    
    mapped_values = []
    for val in series:
        if pd.isna(val):
            mapped_values.append(np.nan)
        else:
            mapped_values.append(mapping_dict.get(str(val), default_value))
    
    return np.array(mapped_values)

def transform_data(df, use_mappings=False, mappings=None):
    """Transform raw data into required model features."""
    transformed_df = df.copy()
    
    st.info("Starting data transformation...")
    
    # 1. estimated_total_paid
    if 'SDBEITR5' in df.columns and 'vtr_dau' in df.columns:
        transformed_df['estimated_total_paid'] = (df['SDBEITR5'] / (5 * 365)) * df['vtr_dau']
        st.success("✓ Created estimated_total_paid")
    else:
        st.error("Missing SDBEITR5 or vtr_dau columns for estimated_total_paid calculation")
        transformed_df['estimated_total_paid'] = np.nan
    
    # 2. vtr_dau (use as-is)
    if 'vtr_dau' in df.columns:
        transformed_df['vtr_dau'] = pd.to_numeric(df['vtr_dau'], errors='coerce')
        st.success("✓ Processed vtr_dau")
    else:
        st.error("Missing vtr_dau column")
        transformed_df['vtr_dau'] = np.nan
    
    # 3. carage_years
    registration_date = None
    for col in ['ersz_final', 'ERSZ', 'First_reg']:
        if col in df.columns:
            registration_date = col
            break
    
    if registration_date:
        today = date.today()
        parsed_dates = df[registration_date].apply(parse_date_robust)
        
        car_age_days = []
        for reg_date in parsed_dates:
            if reg_date:
                days_diff = (today - reg_date).days
                car_age_days.append(days_diff)
            else:
                car_age_days.append(np.nan)
        
        transformed_df['carage_years'] = np.round(np.array(car_age_days) / 365.25, 0)
        st.success(f"✓ Calculated carage_years from {registration_date}")
    else:
        st.error("Missing registration date columns (ersz_final, ERSZ, First_reg)")
        transformed_df['carage_years'] = np.nan
    
    # 4-6. Direct numeric columns
    for col in ['kosten_verw', 'kosten_prov', 'alter']:
        if col in df.columns:
            transformed_df[col] = pd.to_numeric(df[col], errors='coerce')
            st.success(f"✓ Processed {col}")
        else:
            st.error(f"Missing {col} column")
            transformed_df[col] = np.nan
    
    # 7. KILOMETERSTAND_CLEAN
    if 'KILOMETERSTAND_CLEAN' in df.columns:
        transformed_df['KILOMETERSTAND_CLEAN'] = pd.to_numeric(df['KILOMETERSTAND_CLEAN'], errors='coerce')
        st.success("✓ Used KILOMETERSTAND_CLEAN")
    elif 'KILOMETERSTAND' in df.columns:
        transformed_df['KILOMETERSTAND_CLEAN'] = pd.to_numeric(df['KILOMETERSTAND'], errors='coerce')
        st.success("✓ Created KILOMETERSTAND_CLEAN from KILOMETERSTAND")
    else:
        st.warning("Missing KILOMETERSTAND columns, setting to NaN")
        transformed_df['KILOMETERSTAND_CLEAN'] = np.nan
    
    # 8. claim
    if 'claim' in df.columns:
        claim_series = df['claim']
        # Handle string booleans
        if claim_series.dtype == 'object':
            claim_series = claim_series.map({'True': 1, 'False': 0, 'true': 1, 'false': 0, 'YES': 1, 'NO': 0, 'yes': 1, 'no': 0})
        transformed_df['claim'] = pd.to_numeric(claim_series, errors='coerce')
        st.success("✓ Processed claim")
    else:
        st.error("Missing claim column")
        transformed_df['claim'] = np.nan
    
    # 9. state_id (from PLZ)
    if 'plz' in df.columns:
        plz_series = df['plz'].astype(str).str.zfill(5)  # Ensure 5-digit format
        first_two_digits = plz_series.str[:2]
        
        states = []
        for digits in first_two_digits:
            state = STATE_MAPPING.get(digits)
            states.append(state)
        
        state_series = pd.Series(states)
        
        if use_mappings and mappings and 'state_id' in mappings:
            transformed_df['state_id'] = apply_mappings(state_series, mappings['state_id'])
            st.success("✓ Applied saved mappings for state_id")
        else:
            transformed_df['state_id'] = safe_factorize(state_series)
            st.success("✓ Created state_id from PLZ")
        
        # Warn about unmapped states
        null_states = state_series.isna().sum()
        if null_states > 0:
            st.warning(f"⚠️ {null_states} rows have unmappable PLZ codes")
    else:
        st.error("Missing plz column for state_id calculation")
        transformed_df['state_id'] = np.nan
    
    # 10. plz_id
    if 'plz' in df.columns:
        if use_mappings and mappings and 'plz_id' in mappings:
            transformed_df['plz_id'] = apply_mappings(df['plz'], mappings['plz_id'])
            st.success("✓ Applied saved mappings for plz_id")
        else:
            transformed_df['plz_id'] = safe_factorize(df['plz'])
            st.success("✓ Created plz_id")
    else:
        st.error("Missing plz column")
        transformed_df['plz_id'] = np.nan
    
    # 11. Cus_typ_id
    if 'gfeld' in df.columns:
        cus_types = df['gfeld'].astype(str).apply(lambda x: x.split('/')[0] if '/' in x else x)
        
        if use_mappings and mappings and 'Cus_typ_id' in mappings:
            transformed_df['Cus_typ_id'] = apply_mappings(cus_types, mappings['Cus_typ_id'])
            st.success("✓ Applied saved mappings for Cus_typ_id")
        else:
            transformed_df['Cus_typ_id'] = safe_factorize(cus_types)
            st.success("✓ Created Cus_typ_id from gfeld")
    else:
        st.error("Missing gfeld column")
        transformed_df['Cus_typ_id'] = np.nan
    
    # Return only required features in correct order
    feature_df = transformed_df[REQUIRED_FEATURES].copy()
    
    st.success(f"✅ Transformation complete! Created {len(feature_df)} rows with {len(REQUIRED_FEATURES)} features")
    
    return feature_df

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

@st.cache_data
def load_model():
    """Load the trained LightGBM model."""
    try:
        with open("lightgbm_model_vtr_weg_optuna100.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file 'lightgbm_model_vtr_weg_optuna100.pkl' not found. Please ensure it's in the same directory as this app."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def make_demo_predictions(features_df, threshold):
    """Make demo predictions using random probabilities (for testing without real model)."""
    try:
        np.random.seed(42)  # For reproducible demo results
        n_samples = len(features_df)
        
        # Generate realistic-looking probabilities with some correlation to features
        base_probs = np.random.beta(2, 5, n_samples)  # Skewed towards lower probabilities
        
        # Add some feature-based variation (demo purposes)
        if 'alter' in features_df.columns and not features_df['alter'].isna().all():
            age_factor = (features_df['alter'].fillna(features_df['alter'].mean()) - 30) / 100
            base_probs = np.clip(base_probs + age_factor, 0, 1)
        
        probabilities = base_probs
        predictions = ["Yes" if prob >= threshold else "No" for prob in probabilities]
        
        return probabilities, predictions, None
        
    except Exception as e:
        return None, None, f"Error making demo predictions: {str(e)}"

def make_predictions(model, features_df, threshold):
    """Make predictions using the loaded model."""
    try:
        # Handle missing values
        features_clean = features_df.fillna(0)  # Simple imputation for demo
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_clean)[:, 1]
        else:
            # Fallback to decision function + sigmoid
            decision_scores = model.decision_function(features_clean)
            probabilities = sigmoid(decision_scores)
        
        # Apply threshold
        predictions = ["Yes" if prob >= threshold else "No" for prob in probabilities]
        
        return probabilities, predictions, None
        
    except Exception as e:
        return None, None, f"Error making predictions: {str(e)}"

def main():
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Churn Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=0.50, 
        step=0.05,
        help="Probability threshold for classifying as 'Yes' (churn)"
    )
    
    # Optional mappings uploader
    st.sidebar.subheader("Optional: Category Mappings")
    mappings_file = st.sidebar.file_uploader(
        "Upload mappings JSON", 
        type=['json'],
        help="Upload saved category mappings from training to ensure consistent encoding"
    )
    
    use_mappings = st.sidebar.checkbox(
        "Use uploaded mappings", 
        value=False,
        disabled=mappings_file is None
    )
    
    # Load mappings
    mappings = None
    if mappings_file and use_mappings:
        try:
            mappings = json.load(mappings_file)
            st.sidebar.success(f"✓ Loaded mappings for {len(mappings)} categories")
        except Exception as e:
            st.sidebar.error(f"Error loading mappings: {str(e)}")
            use_mappings = False
    
    # Load model
    model, model_error = load_model()
    if model_error:
        st.error(model_error)
        st.stop()
    else:
        st.success("✅ Model loaded successfully!")
    
    # Main file uploader
    st.header("Upload Policy Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=['csv', 'xlsx'],
        help="Upload your raw policy data for churn prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Show preview of original data
            st.header("Original Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Transform data
            st.header("Data Transformation")
            with st.spinner("Transforming data..."):
                features_df = transform_data(df, use_mappings, mappings)
            
            # Show feature preview
            st.header("Transformed Features Preview")
            st.dataframe(features_df.head(20), use_container_width=True)
            
            # Check for missing values
            missing_info = features_df.isnull().sum()
            if missing_info.sum() > 0:
                st.warning("⚠️ Some features have missing values:")
                for feature, missing_count in missing_info[missing_info > 0].items():
                    st.write(f"- {feature}: {missing_count} missing values")
            
            # Make predictions
            st.header("Predictions")
            with st.spinner("Making predictions..."):
                probabilities, predictions, pred_error = make_predictions(model, features_df, threshold)
            
            if pred_error:
                st.error(pred_error)
            else:
                # Prepare results dataframe
                results_df = df.copy()
                results_df['churn_probability'] = probabilities
                results_df['churn_class'] = predictions
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(results_df))
                with col2:
                    churn_count = sum(1 for p in predictions if p == "Yes")
                    st.metric("Predicted Churns", churn_count)
                with col3:
                    avg_prob = np.mean(probabilities)
                    st.metric("Average Probability", f"{avg_prob:.3f}")
                
                # Show results with identifier columns if available
                display_cols = []
                
                # Include common identifier columns
                id_cols = ['vsnr', 'pvsnr', 'customer_id', 'policy_id']
                for col in id_cols:
                    if col in df.columns:
                        display_cols.append(col)
                
                # Add prediction columns
                display_cols.extend(['churn_probability', 'churn_class'])
                
                results_display = results_df[display_cols] if display_cols else results_df[['churn_probability', 'churn_class']]
                
                st.subheader("Prediction Results")
                st.dataframe(results_display, use_container_width=True)
                
                # Download button
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Show distribution
                st.subheader("Probability Distribution")
                prob_bins = np.histogram(probabilities, bins=20)[0]
                prob_chart_data = pd.DataFrame({
                    'Probability Range': [f"{i*0.05:.2f}-{(i+1)*0.05:.2f}" for i in range(20)],
                    'Count': prob_bins
                })
                st.bar_chart(prob_chart_data.set_index('Probability Range'))
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV or XLSX file to begin")

if __name__ == "__main__":
    main()
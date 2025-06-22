from flask import Flask, request, jsonify
import joblib
import lightgbm as lgb  # must be imported so joblib knows how to reconstruct the model

# ✅ Load model with joblib
model = joblib.load('enhanced_tuned_lightgbm_model.pkl')

app = Flask(__name__)

selected_features = [ 
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = [data[feat] for feat in selected_features]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

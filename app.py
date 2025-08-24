from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the trained model and the encoder
try:
    model = joblib.load('lung_disease_model.joblib')
    encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    print("Error: Model or encoder file not found. Make sure 'lung_disease_model.joblib' and 'label_encoder.joblib' are in the same directory.")
    model = None
    encoder = None

@app.route('/')
def home():
    # This route now correctly renders the index.html template.
    return render_template('index.html')

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    if not model or not encoder:
        return jsonify({'error': 'Model not loaded.'}), 500

    data = request.get_json()
    
    feature_names = [
        "index", "Age", "Gender", "Air Pollution", "Alcohol use",
        "Dust Allergy", "OccuPational Hazards", "Genetic Risk",
        "chronic Lung Disease", "Balanced Diet", "Obesity", "Smoking",
        "Passive Smoker", "Chest Pain", "Coughing of Blood",
        "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
        "Swallowing Difficulty", "Clubbing of Finger Nails",
        "Frequent Cold", "Dry Cough", "Snoring"
    ]
    
    try:
        input_data = [int(data[name]) for name in feature_names]
        input_df = pd.DataFrame([input_data], columns=feature_names)
    except (ValueError, KeyError) as e:
        return jsonify({'error': f"Error processing input data: {e}"}), 400

    dmatrix = xgb.DMatrix(input_df)
    
    prediction_numeric = model.predict(dmatrix)
    
    prediction_label = encoder.inverse_transform(prediction_numeric.astype(int))

    return jsonify({'prediction': prediction_label[0]})

if __name__ == '__main__':
    app.run(debug=True)